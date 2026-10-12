/****************************************************************************
 * @file      bench_cwprf_mqrpmt.cpp
 * @brief     Performance benchmark suite for cwPRF-based mqRPMT protocol.
 *
 * Benchmarked configurations
 *   - Secp256r1 (compressed,   33 B/pt) + BloomFilter
 *   - Secp256r1 (compressed,   33 B/pt) + PlainSet
 *   - Secp256r1 (uncompressed, 65 B/pt) + BloomFilter
 *   - Secp256r1 (uncompressed, 65 B/pt) + PlainSet
 *   - X25519    (always 32 B/pt)        + BloomFilter
 *   - X25519    (always 32 B/pt)        + PlainSet
 *
 * Dataset
 *   Server : 2^20 elements
 *   Client : 2^20 elements
 *   Intersection : 50%
 *
 * Author: Yu Chen
 ****************************************************************************/

#include <taihang/mpc/pso/cwprf_mqrpmt.hpp>
#include <taihang/common/config.hpp>
#include <taihang/common/logger.hpp>
#include <taihang/common/bench_setting.hpp>
#include <taihang/crypto/prg.hpp>
#include <taihang/crypto/x25519_simd.hpp>
#include <taihang/system/cpu.hpp>
#include <openssl/obj_mac.h>
#include <chrono>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace taihang;
using namespace taihang::mpc::cwprf_mqrpmt;

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

//==============================================================================
// Dataset
//==============================================================================

struct Dataset
{
    std::vector<Block> vec_y;
    std::vector<Block> vec_x;
    size_t intersection_len;
};

static Dataset make_dataset(size_t server_len,
                            size_t client_len)
{
    Block seed_block =
        make_block(0x123456789ABCDEF0ULL,
                   0x0FEDCBA987654321ULL);

    prg::Seed seed = prg::set_seed(&seed_block, 0);

    Dataset ds;

    ds.vec_y.resize(server_len);
    ds.vec_x.resize(client_len);

    ds.intersection_len = client_len / 2;

    prg::gen_random_blocks(seed, ds.vec_y.data(), server_len);

    for (size_t i = 0; i < client_len; ++i)
    {
        if (i < ds.intersection_len)
            ds.vec_x[i] = ds.vec_y[i];
        else
            ds.vec_x[i] = make_block(i, 0xDEADBEEFADDEULL);
    }

    return ds;
}


//==============================================================================
// Benchmark Result
//==============================================================================

struct BenchResult
{
    double server_ms;
    double client_ms;
    double wall_ms;
    size_t matches;
};


//==============================================================================
// Run one benchmark
//==============================================================================

static BenchResult run_once(const PublicParameters& pp,
                            const Dataset& ds,
                            uint16_t port)
{
    const std::string address = "127.0.0.1";

    auto wall_begin = Clock::now();

    auto server_future =
        std::async(std::launch::async, [&]() {
            net::NetIO io("server", address, port);
            auto t0   = Clock::now();
            auto bits = server(io, pp, ds.vec_y);
            double ms = Ms(Clock::now() - t0).count();
            return std::make_pair(std::move(bits), ms);
        });

    auto client_future =
        std::async(std::launch::async, [&]() {
            net::NetIO io("client", address, port);
            auto t0 = Clock::now();
            client(io, pp, ds.vec_x);
            return Ms(Clock::now() - t0).count();
        });

    double client_ms           = client_future.get();
    auto [bits, server_ms]     = server_future.get();
    double wall_ms             = Ms(Clock::now() - wall_begin).count();

    size_t matches = 0;
    for (auto b : bits) matches += b;

    return { server_ms, client_ms, wall_ms, matches };
}


//==============================================================================
// Pretty Printing
//==============================================================================

static double throughput_kelem_per_sec(size_t n, double ms)
{
    return n / (ms / 1000.0) / 1000.0;
}

static void print_header(const std::string& label,
                         size_t server_len, size_t client_len,
                         size_t log_server, size_t log_client,
                         int curve_id)
{
    std::cout << "\n"
        << "================================================================================\n"
        << label << "\n"
        << "--------------------------------------------------------------------------------\n"
        << "Server Set         : 2^" << log_server << " = " << server_len << "\n"
        << "Client Set         : 2^" << log_client << " = " << client_len << "\n"
        << "Intersection Ratio : 50%\n";
    if (curve_id == NID_X25519) {
        std::cout << "X25519 Backend     : "
                  << crypto::x25519_simd::backend_name(crypto::x25519_simd::Backend::Auto)
                  << "\n";
    }
    std::cout << "================================================================================\n";
}

static void print_result(const BenchResult& r,
                         size_t server_len, size_t client_len,
                         size_t expected_intersection)
{
    double server_tp = throughput_kelem_per_sec(server_len, r.server_ms);
    double client_tp = throughput_kelem_per_sec(client_len, r.client_ms);

    std::cout << std::fixed << std::setprecision(2)
        << "Wall Time      : " << r.wall_ms   / 1000.0 << " s\n"
        << "Server Time    : " << r.server_ms / 1000.0 << " s    (" << server_tp << " Kelem/s)\n"
        << "Client Time    : " << r.client_ms / 1000.0 << " s    (" << client_tp << " Kelem/s)\n"
        << "Intersection   : " << r.matches << " / " << expected_intersection
        << (r.matches >= expected_intersection ? "  [PASS]" : "  [FAIL]") << "\n";
}

static void print_summary(
    const std::vector<std::pair<std::string, BenchResult>>& results,
    size_t server_len, size_t client_len,
    size_t log_server, size_t log_client)
{
    constexpr int LABEL_W = 38;

    std::cout << "\n\n"
        << "========================================================================================================\n"
        << "                                  Taihang cwPRF-mqRPMT Benchmark\n"
        << "========================================================================================================\n"
        << "Dataset : 2^" << log_server << " (" << server_len << ") server elements"
        << "    2^" << log_client << " (" << client_len << ") client elements"
        << "    50% intersection\n"
        << "Threads : " << config::thread_num << " per party  ("
        << config::thread_num * 2 << " concurrent total, single-machine loopback)\n"
        << "--------------------------------------------------------------------------------------------------------\n"
        << std::left  << std::setw(LABEL_W) << "Configuration"
        << std::right << std::setw(12) << "Server(s)"
                      << std::setw(12) << "Client(s)"
                      << std::setw(12) << "Wall(s)"
                      << std::setw(14) << "Kelem/s"
        << "\n"
        << "--------------------------------------------------------------------------------------------------------\n";

    for (const auto& [label, r] : results)
    {
        double tp = throughput_kelem_per_sec(server_len, r.server_ms);
        std::cout
            << std::left  << std::setw(LABEL_W) << label
            << std::right << std::fixed << std::setprecision(2)
            << std::setw(12) << r.server_ms / 1000.0
            << std::setw(12) << r.client_ms / 1000.0
            << std::setw(12) << r.wall_ms   / 1000.0
            << std::setw(14) << tp
            << "\n";
    }

    // ── Compression speedup analysis (only for Secp256r1 rows) ──────────────
    // rows 0,1 = compressed (bloom, plain)
    // rows 2,3 = uncompressed (bloom, plain)
    if (results.size() >= 4)
    {
        double comp_bloom_s   = results[0].second.server_ms;
        double comp_plain_s   = results[1].second.server_ms;
        double uncomp_bloom_s = results[2].second.server_ms;
        double uncomp_plain_s = results[3].second.server_ms;

        std::cout
            << "--------------------------------------------------------------------------------------------------------\n"
            << std::fixed << std::setprecision(2)
            << "Secp256r1 Compression Speedup (compressed vs uncompressed)\n"
            << "  BloomFilter : " << uncomp_bloom_s / comp_bloom_s << "x\n"
            << "  PlainSet    : " << uncomp_plain_s / comp_plain_s << "x\n";
    }

    // ── BloomFilter vs PlainSet speedup ─────────────────────────────────────
    if (results.size() >= 6)
    {
        double secp_comp_speedup   = results[1].second.server_ms / results[0].second.server_ms;
        double secp_uncomp_speedup = results[3].second.server_ms / results[2].second.server_ms;
        double x25519_speedup      = results[5].second.server_ms / results[4].second.server_ms;

        std::cout
            << "--------------------------------------------------------------------------------------------------------\n"
            << "BloomFilter Speedup over PlainSet\n"
            << "  Secp256r1 (compressed)   : " << secp_comp_speedup   << "x\n"
            << "  Secp256r1 (uncompressed) : " << secp_uncomp_speedup << "x\n"
            << "  X25519                   : " << x25519_speedup       << "x\n";
    }

    std::cout
        << "========================================================================================================\n";
}


//==============================================================================
// Main
//==============================================================================

int main()
{
    constexpr size_t kLogServerLen = 20;
    constexpr size_t kLogClientLen = 20;

    // Six configurations — each needs a distinct port
    constexpr uint16_t kPortSecpCompBloom   = 12350;
    constexpr uint16_t kPortSecpCompPlain   = 12351;
    constexpr uint16_t kPortSecpUncompBloom = 12352;
    constexpr uint16_t kPortSecpUncompPlain = 12353;
    constexpr uint16_t kPortX25519Bloom     = 12354;
    constexpr uint16_t kPortX25519Plain     = 12355;

    const size_t server_len = 1ULL << kLogServerLen;
    const size_t client_len = 1ULL << kLogClientLen;

    //----------------------------------------------------------------------
    // Generate dataset (once, shared across all runs)
    //----------------------------------------------------------------------

    std::cout << "Generating benchmark dataset ...\n";
    auto t0 = Clock::now();
    Dataset ds = make_dataset(server_len, client_len);
    std::cout << "Dataset ready in "
              << std::fixed << std::setprecision(2)
              << Ms(Clock::now() - t0).count() << " ms\n";



    //==============================================================================
    // Thread configuration
    //==============================================================================

    constexpr BenchmarkMode kBenchmarkMode = BenchmarkMode::SingleMachine; 
    thread_configuration(kBenchmarkMode); 

    std::vector<std::pair<std::string, BenchResult>> summary;

    //======================================================================
    // Secp256r1 + Compressed (33 B/point) + BloomFilter
    //======================================================================
    {
        config::use_point_compression = true;   // ← 33 bytes/point

        print_header("Benchmark: Secp256r1 [compressed, 33 bytes/point] + BloomFilter",
                     server_len, client_len, kLogServerLen, kLogClientLen,
                     NID_X9_62_prime256v1);

        auto pp = setup(415, kLogServerLen, kLogClientLen, MembershipMode::BloomFilter, 40);
        auto r  = run_once(pp, ds, kPortSecpCompBloom);
        print_result(r, server_len, client_len, ds.intersection_len);
        summary.emplace_back("Secp256r1 [comp]   + BloomFilter", r);
    }

    //======================================================================
    // Secp256r1 + Compressed (33 B/point) + PlainSet
    //======================================================================
    {
        config::use_point_compression = true;

        print_header("Benchmark: Secp256r1 [compressed, 33 bytes/point] + PlainSet",
                     server_len, client_len, kLogServerLen, kLogClientLen,
                     NID_X9_62_prime256v1);

        auto pp = setup(415, kLogServerLen, kLogClientLen, MembershipMode::PlainSet);
        auto r  = run_once(pp, ds, kPortSecpCompPlain);
        print_result(r, server_len, client_len, ds.intersection_len);
        summary.emplace_back("Secp256r1 [comp]   + PlainSet   ", r);
    }

    //======================================================================
    // Secp256r1 + Uncompressed (65 B/point) + BloomFilter
    //======================================================================
    {
        config::use_point_compression = false;  // ← 65 bytes/point

        print_header("Benchmark: Secp256r1 [uncompressed, 65 bytes/point] + BloomFilter",
                     server_len, client_len, kLogServerLen, kLogClientLen,
                     NID_X9_62_prime256v1);

        auto pp = setup(415, kLogServerLen, kLogClientLen, MembershipMode::BloomFilter, 40);
        auto r  = run_once(pp, ds, kPortSecpUncompBloom);
        print_result(r, server_len, client_len, ds.intersection_len);
        summary.emplace_back("Secp256r1 [uncomp] + BloomFilter", r);
    }

    //======================================================================
    // Secp256r1 + Uncompressed (65 B/point) + PlainSet
    //======================================================================
    {
        config::use_point_compression = false;

        print_header("Benchmark: Secp256r1 [uncompressed, 65 bytes/point] + PlainSet",
                     server_len, client_len, kLogServerLen, kLogClientLen,
                     NID_X9_62_prime256v1);

        auto pp = setup(415, kLogServerLen, kLogClientLen, MembershipMode::PlainSet);
        auto r  = run_once(pp, ds, kPortSecpUncompPlain);
        print_result(r, server_len, client_len, ds.intersection_len);
        summary.emplace_back("Secp256r1 [uncomp] + PlainSet   ", r);
    }

    // Restore default for X25519 runs (irrelevant but tidy)
    config::use_point_compression = true;

    //======================================================================
    // X25519 + BloomFilter  (always 32 B/point, no compression switch)
    //======================================================================
    {
        print_header("Benchmark: X25519 [32 bytes/point] + BloomFilter",
                     server_len, client_len, kLogServerLen, kLogClientLen,
                     NID_X25519);

        auto pp = setup(NID_X25519, kLogServerLen, kLogClientLen, MembershipMode::BloomFilter, 40);
        auto r  = run_once(pp, ds, kPortX25519Bloom);
        print_result(r, server_len, client_len, ds.intersection_len);
        summary.emplace_back("X25519             + BloomFilter", r);
    }

    //======================================================================
    // X25519 + PlainSet
    //======================================================================
    {
        print_header("Benchmark: X25519 [32 bytes/point] + PlainSet",
                     server_len, client_len, kLogServerLen, kLogClientLen,
                     NID_X25519);

        auto pp = setup(NID_X25519, kLogServerLen, kLogClientLen, MembershipMode::PlainSet);
        auto r  = run_once(pp, ds, kPortX25519Plain);
        print_result(r, server_len, client_len, ds.intersection_len);
        summary.emplace_back("X25519             + PlainSet   ", r);
    }

    //----------------------------------------------------------------------
    // Final summary table
    //----------------------------------------------------------------------

    print_summary(summary, server_len, client_len, kLogServerLen, kLogClientLen);

    return 0;
}

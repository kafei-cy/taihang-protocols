/****************************************************************************
 * @file      test_alsz_ote.cpp
 * @brief     GTest suite for ALSZ OT Extension (ALSZ OTE).
 * @author    Yu Chen
 *****************************************************************************/

#include <gtest/gtest.h>
#include <taihang/mpc/ot/alsz_ote.hpp>
#include <taihang/crypto/prg.hpp>
#include <openssl/obj_mac.h>
#include <thread>
#include <vector>

using namespace taihang;
using namespace taihang::mpc::alsz_ote;

// ============================================================
// Protocol Test Fixture
// ============================================================

class ALSZOteTest : public ::testing::Test {
protected:
    void SetUp() override {
        kCurveId   = NID_X9_62_prime256v1; // NIST P-256 for base OTs
        kBaseLen   = 128;                 // Security parameter \kappa (number of base OTs)
        kExtendLen = 1024;                // Number of extended OTs (must be multiple of 128)
        kPort      = 12347;
        kAddress   = "127.0.0.1";

        pp = setup(kCurveId, kBaseLen);

        // Pre-generate random structural data context blocks
        prg::Seed seed = prg::set_seed(nullptr, 0);
        vec_m0 = prg::gen_random_blocks(seed, kExtendLen);
        vec_m1 = prg::gen_random_blocks(seed, kExtendLen);
        
        vec_selection_bit.resize(kExtendLen);
        for (size_t i = 0; i < kExtendLen; ++i) {
            vec_selection_bit[i] = static_cast<uint8_t>(i % 2);
        }
    }

    int kCurveId;
    size_t kBaseLen;
    size_t kExtendLen;
    uint16_t kPort;
    std::string kAddress;

    PublicParameters pp;
    std::vector<Block> vec_m0;
    std::vector<Block> vec_m1;
    std::vector<uint8_t> vec_selection_bit;
};

// ============================================================
// 1. Parameter Parameterization & Structural Validity
// ============================================================

TEST_F(ALSZOteTest, Setup_Parameter_Fields) {
    EXPECT_FALSE(pp.malicious);
    EXPECT_EQ(pp.base_len, kBaseLen);
    EXPECT_EQ(pp.base_ot_pp.curve_id, kCurveId);
    EXPECT_NE(pp.base_ot_pp.group_ctx, nullptr);
}

// ============================================================
// 2. Core Protocol Execution Pipeline (Standard Two-Sided)
// ============================================================

TEST_F(ALSZOteTest, Execute_Standard_Protocol_Roundtrip) {
    std::vector<Block> vec_result;

    // Run Sender instance inside an isolated worker background context ("server")
    std::thread sender_worker([&]() {
        net::NetIO io_sender("server", kAddress, kPort);
        sender<BlockPolicy>(io_sender, pp, vec_m0, vec_m1, kExtendLen);
    });

    // Run Receiver instance on the execution primary loop context ("client")
    std::thread receiver_worker([&]() {
        net::NetIO io_receiver("client", kAddress, kPort);
        vec_result = receiver<BlockPolicy>(io_receiver, pp, vec_selection_bit, kExtendLen);
    });

    sender_worker.join();
    receiver_worker.join();

    // Verify consistency criteria across all extended OT choices
    ASSERT_EQ(vec_result.size(), kExtendLen);
    for (size_t i = 0; i < kExtendLen; ++i) {
        if (vec_selection_bit[i] == 0) {
            EXPECT_EQ(vec_result[i], vec_m0[i]) << "Mismatch at extended index: " << i << " for choice 0";
        } else {
            EXPECT_EQ(vec_result[i], vec_m1[i]) << "Mismatch at extended index: " << i << " for choice 1";
        }
    }
}

// ============================================================
// 3. Serialization Validation
// ============================================================

TEST_F(ALSZOteTest, Serialization_PublicParameters_RoundTrip) {
    std::ostringstream oss;
    oss << pp;

    PublicParameters pp_reconstructed;
    std::istringstream iss(oss.str());
    iss >> pp_reconstructed;

    EXPECT_EQ(pp.malicious, pp_reconstructed.malicious);
    EXPECT_EQ(pp.base_len, pp_reconstructed.base_len);
    EXPECT_EQ(pp.base_ot_pp.curve_id, pp_reconstructed.base_ot_pp.curve_id);
}
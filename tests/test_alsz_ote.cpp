/****************************************************************************
 * @file      test_alsz_ote.cpp
 * @brief     GTest suite for ALSZ OT Extension (ALSZ OTE).
 * @author    Yu Chen
 *****************************************************************************/

#include <gtest/gtest.h>
#include <taihang/mpc/ot/alsz_ote.hpp>
#include <taihang/crypto/prg.hpp>
#include <openssl/obj_mac.h>
#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>
#include <tuple>
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

// ============================================================
// 4. Compatibility with the Original Row Hash
// ============================================================

namespace {

// Adapt only the wire layout. The reference still expands and transposes a
// complete matrix, independently of the production streaming implementation.
// At base_len == 128, P is sent as compact 2048-row tiles, each in column order.
std::vector<Block> reference_p_wire(const std::vector<Block>& matrix,
                                   size_t base_len, bool encode) {
    if (base_len != 128) {
        return matrix;
    }
    const size_t column_blocks = matrix.size() / base_len;
    std::vector<Block> result(matrix.size());
    size_t wire_offset = 0;
    for (size_t first = 0; first < column_blocks; first += 16) {
        const size_t count = std::min(size_t{16}, column_blocks - first);
        for (size_t column = 0; column < base_len; ++column) {
            for (size_t i = 0; i < count; ++i) {
                const size_t matrix_offset = column * column_blocks + first + i;
                if (encode) {
                    result[wire_offset++] = matrix[matrix_offset];
                } else {
                    result[matrix_offset] = matrix[wire_offset++];
                }
            }
        }
    }
    return result;
}

// Keep the original scalar row hash as an independent peer. A roundtrip
// between two optimized peers alone cannot detect the same hash change on both.
std::vector<Block> reference_random_receiver(net::NetIO& io,
        const PublicParameters& pp, const std::vector<uint8_t>& choices) {
    const size_t row_num = choices.size();
    const size_t column_blocks = row_num / 128;
    const size_t row_blocks = pp.base_len / 128;
    const Block salt = make_block(0x123456789abcdef0ULL, 0x0fedcba987654321ULL);
    auto seed = prg::set_seed(&salt, 0);
    auto t_seeds = prg::gen_random_blocks(seed, pp.base_len);
    auto u_seeds = prg::gen_random_blocks(seed, pp.base_len);
    mpc::np_ot::sender(io, pp.base_ot_pp, t_seeds, u_seeds, pp.base_len);

    std::vector<Block> packed_choices(column_blocks);
    pack_bits_to_blocks(choices.data(), row_num, packed_choices.data(), column_blocks);
    std::vector<Block> matrix_t(column_blocks * pp.base_len);
    std::vector<Block> matrix_p(matrix_t.size());
    for (size_t j = 0; j < pp.base_len; ++j) {
        prg::reset_seed(seed, &t_seeds[j], 0);
        auto t_column = prg::gen_random_blocks(seed, column_blocks);
        prg::reset_seed(seed, &u_seeds[j], 0);
        auto u_column = prg::gen_random_blocks(seed, column_blocks);
        auto p_column = t_column ^ u_column ^ packed_choices;
        std::memcpy(matrix_t.data() + j * column_blocks,
                    t_column.data(), column_blocks * sizeof(Block));
        std::memcpy(matrix_p.data() + j * column_blocks,
                    p_column.data(), column_blocks * sizeof(Block));
    }
    io.send(reference_p_wire(matrix_p, pp.base_len, true));

    std::vector<Block> transpose(matrix_t.size());
    bit_matrix_transpose(reinterpret_cast<uint8_t*>(matrix_t.data()), pp.base_len,
                         row_num, reinterpret_cast<uint8_t*>(transpose.data()));
    std::vector<Block> keys(row_num);
    for (size_t i = 0; i < row_num; ++i) {
        std::vector<Block> row(transpose.begin() + i * row_blocks,
                               transpose.begin() + (i + 1) * row_blocks);
        keys[i] = aes::hash_blocks_to_block(row);
    }
    return keys;
}

std::pair<std::vector<Block>, std::vector<Block>> reference_random_sender(
        net::NetIO& io, const PublicParameters& pp, size_t row_num) {
    const size_t column_blocks = row_num / 128;
    const size_t row_blocks = pp.base_len / 128;
    const Block salt = make_block(0x0fedcba987654321ULL, 0x123456789abcdef0ULL);
    auto seed = prg::set_seed(&salt, 0);
    auto choices = prg::gen_random_bits(seed, pp.base_len);
    auto q_seeds = mpc::np_ot::receiver(io, pp.base_ot_pp, choices, pp.base_len);

    std::vector<Block> matrix_q(column_blocks * pp.base_len);
    for (size_t j = 0; j < pp.base_len; ++j) {
        prg::reset_seed(seed, &q_seeds[j], 0);
        auto column = prg::gen_random_blocks(seed, column_blocks);
        std::memcpy(matrix_q.data() + j * column_blocks,
                    column.data(), column_blocks * sizeof(Block));
    }
    std::vector<Block> matrix_p(matrix_q.size());
    io.recv(matrix_p);
    matrix_p = reference_p_wire(matrix_p, pp.base_len, false);
    for (size_t j = 0; j < pp.base_len; ++j) {
        for (size_t i = 0; i < column_blocks; ++i) {
            if (choices[j] == 1) {
                matrix_q[j * column_blocks + i] ^= matrix_p[j * column_blocks + i];
            }
        }
    }
    std::vector<Block> transpose(matrix_q.size());
    bit_matrix_transpose(reinterpret_cast<uint8_t*>(matrix_q.data()), pp.base_len,
                         row_num, reinterpret_cast<uint8_t*>(transpose.data()));
    std::vector<Block> packed_choices(row_blocks);
    pack_bits_to_blocks(choices.data(), pp.base_len, packed_choices.data(), row_blocks);

    std::vector<Block> k0(row_num), k1(row_num);
    for (size_t i = 0; i < row_num; ++i) {
        std::vector<Block> row(transpose.begin() + i * row_blocks,
                               transpose.begin() + (i + 1) * row_blocks);
        k0[i] = aes::hash_blocks_to_block(row);
        k1[i] = aes::hash_blocks_to_block(row ^ packed_choices);
    }
    return {std::move(k0), std::move(k1)};
}

} // namespace

class ALSZOteHashTest : public ALSZOteTest,
                       public ::testing::WithParamInterface<std::tuple<size_t, int, size_t, int>> {
protected:
    void SetUp() override {
        ALSZOteTest::SetUp();
        old_thread_num = config::thread_num;
        int choice_pattern;
        std::tie(kBaseLen, config::thread_num, kExtendLen, choice_pattern) = GetParam();
        pp = setup(kCurveId, kBaseLen);
        const Block salt = make_block(0x3141592653589793ULL, 0x2718281828459045ULL);
        auto seed = prg::set_seed(&salt, 0);
        vec_m0 = prg::gen_random_blocks(seed, kExtendLen);
        vec_m1 = prg::gen_random_blocks(seed, kExtendLen);
        vec_selection_bit = prg::gen_random_bits(seed, kExtendLen);
        for (size_t i = 0; i < kExtendLen; ++i) {
            if (choice_pattern < 2) {
                vec_selection_bit[i] = static_cast<uint8_t>(choice_pattern);
            } else if (choice_pattern == 2) {
                vec_selection_bit[i] = static_cast<uint8_t>((i % 5) < 2);
            }
        }
    }

    void TearDown() override {
        config::thread_num = old_thread_num;
    }

    void expect_selected_keys(const std::pair<std::vector<Block>, std::vector<Block>>& keys,
                              const std::vector<Block>& selected) {
        ASSERT_EQ(keys.first.size(), kExtendLen);
        ASSERT_EQ(keys.second.size(), kExtendLen);
        ASSERT_EQ(selected.size(), kExtendLen);
        for (size_t i = 0; i < kExtendLen; ++i) {
            EXPECT_EQ(selected[i], vec_selection_bit[i] ? keys.second[i] : keys.first[i])
                << "Mismatch at extended index: " << i;
        }
    }

    int old_thread_num = 0;
};

TEST_P(ALSZOteHashTest, Sender_Matches_Original_Row_Hash) {
    std::pair<std::vector<Block>, std::vector<Block>> keys;
    std::vector<Block> selected;
    std::thread sender_worker([&]() {
        net::NetIO io("server", kAddress, kPort);
        keys = random_sender(io, pp, kExtendLen);
    });
    std::thread receiver_worker([&]() {
        net::NetIO io("client", kAddress, kPort);
        selected = reference_random_receiver(io, pp, vec_selection_bit);
    });
    sender_worker.join();
    receiver_worker.join();
    expect_selected_keys(keys, selected);
}

TEST_P(ALSZOteHashTest, Receiver_Matches_Original_Row_Hash) {
    std::pair<std::vector<Block>, std::vector<Block>> keys;
    std::vector<Block> selected;
    std::thread sender_worker([&]() {
        net::NetIO io("server", kAddress, kPort);
        keys = reference_random_sender(io, pp, kExtendLen);
    });
    std::thread receiver_worker([&]() {
        net::NetIO io("client", kAddress, kPort);
        selected = random_receiver(io, pp, vec_selection_bit, kExtendLen);
    });
    sender_worker.join();
    receiver_worker.join();
    expect_selected_keys(keys, selected);
}

TEST_P(ALSZOteHashTest, Execute_Bytes_Protocol_Roundtrip) {
    // Exercise messages that are longer than one block and not block-aligned.
    std::vector<BytesPolicy::Message> m0(kExtendLen), m1(kExtendLen), result;
    for (size_t i = 0; i < kExtendLen; ++i) {
        m0[i].resize(31);
        m1[i].resize(31);
        for (size_t j = 0; j < 31; ++j) {
            m0[i][j] = static_cast<uint8_t>(i + j);
            m1[i][j] = static_cast<uint8_t>(i + j + 1);
        }
    }
    std::thread sender_worker([&]() {
        net::NetIO io("server", kAddress, kPort);
        sender<BytesPolicy>(io, pp, m0, m1, kExtendLen);
    });
    std::thread receiver_worker([&]() {
        net::NetIO io("client", kAddress, kPort);
        result = receiver<BytesPolicy>(io, pp, vec_selection_bit, kExtendLen);
    });
    sender_worker.join();
    receiver_worker.join();
    ASSERT_EQ(result.size(), kExtendLen);
    for (size_t i = 0; i < kExtendLen; ++i) {
        EXPECT_EQ(result[i], vec_selection_bit[i] ? m1[i] : m0[i]) << "Index: " << i;
    }
}

TEST_P(ALSZOteHashTest, Execute_Onesided_Protocol_Preserves_Order) {
    std::vector<Block> result, expected;
    for (size_t i = 0; i < kExtendLen; ++i) {
        if (vec_selection_bit[i] == 1) {
            expected.push_back(vec_m1[i]);
        }
    }
    std::thread sender_worker([&]() {
        net::NetIO io("server", kAddress, kPort);
        onesided_sender<BlockPolicy>(io, pp, vec_m1, kExtendLen);
    });
    std::thread receiver_worker([&]() {
        net::NetIO io("client", kAddress, kPort);
        result = onesided_receiver<BlockPolicy>(io, pp, vec_selection_bit, kExtendLen);
    });
    sender_worker.join();
    receiver_worker.join();
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(result[i], expected[i]) << "Selected index: " << i;
    }
}

// Choices: 0/1 are constant, 2 preserves the periodic baseline, 3 uses a fixed PRG.
// Check a short tile, an exact tile, a partial second tile and a window tail.
INSTANTIATE_TEST_SUITE_P(BaseLengthThreadsRowsAndChoices, ALSZOteHashTest,
    ::testing::Values(std::make_tuple(size_t{128}, 1, size_t{384}, 2),
                      std::make_tuple(size_t{128}, 4, size_t{384}, 2),
                      std::make_tuple(size_t{256}, 1, size_t{384}, 2),
                      std::make_tuple(size_t{256}, 4, size_t{384}, 2),
                      std::make_tuple(size_t{128}, 1, size_t{128}, 0),
                      std::make_tuple(size_t{128}, 4, size_t{2048}, 1),
                      std::make_tuple(size_t{128}, 1, size_t{2176}, 3),
                      std::make_tuple(size_t{128}, 4, size_t{32896}, 3)));

// ============================================================
// 5. Buffer Reuse and Original Ciphertext Framing
// ============================================================

class ALSZOteBufferTest : public ALSZOteTest,
                         public ::testing::WithParamInterface<std::tuple<size_t, int, int, size_t>> {
protected:
    void SetUp() override {
        ALSZOteTest::SetUp();
        old_thread_num = config::thread_num;
        int choice_pattern;
        std::tie(kBaseLen, config::thread_num, choice_pattern, kExtendLen) = GetParam();
        // Exercise both the ciphertext buffer and the matrix window boundaries.
        pp = setup(kCurveId, kBaseLen);

        const Block salt = make_block(0x3141592653589793ULL, 0x2718281828459045ULL);
        auto seed = prg::set_seed(&salt, 0);
        vec_m0 = prg::gen_random_blocks(seed, kExtendLen);
        vec_m1 = prg::gen_random_blocks(seed, kExtendLen);
        vec_selection_bit = prg::gen_random_bits(seed, kExtendLen);
        for (size_t i = 0; i < kExtendLen; ++i) {
            if (choice_pattern < 2) {
                vec_selection_bit[i] = static_cast<uint8_t>(choice_pattern);
            } else if (choice_pattern == 2) {
                vec_selection_bit[i] = static_cast<uint8_t>((i % 5) < 2);
            }
        }
    }

    void TearDown() override {
        config::thread_num = old_thread_num;
    }

    // Retain independent keys, ciphertexts and whole-vector network calls on
    // this peer, including the original C0-then-C1 wire order.
    void reference_sender(net::NetIO& io, bool onesided) {
        auto keys = reference_random_sender(io, pp, kExtendLen);
        std::vector<Block> c1(kExtendLen);
        for (size_t i = 0; i < kExtendLen; ++i) {
            c1[i] = keys.second[i] ^ vec_m1[i];
        }
        if (!onesided) {
            std::vector<Block> c0(kExtendLen);
            for (size_t i = 0; i < kExtendLen; ++i) {
                c0[i] = keys.first[i] ^ vec_m0[i];
            }
            io.send(c0);
        }
        io.send(c1);
    }

    std::vector<Block> reference_receiver(net::NetIO& io, bool onesided) {
        auto keys = reference_random_receiver(io, pp, vec_selection_bit);
        std::vector<Block> c0, c1(kExtendLen), result;
        if (!onesided) {
            c0.resize(kExtendLen);
            io.recv(c0);
        }
        io.recv(c1);
        for (size_t i = 0; i < kExtendLen; ++i) {
            if (vec_selection_bit[i] == 1) {
                result.push_back(c1[i] ^ keys[i]);
            } else if (!onesided) {
                result.push_back(c0[i] ^ keys[i]);
            }
        }
        return result;
    }

    void run_with_reference_peer(bool optimized_sender, bool onesided) {
        std::vector<Block> result;
        std::thread sender_worker([&]() {
            net::NetIO io("server", kAddress, kPort);
            if (!optimized_sender) {
                reference_sender(io, onesided);
            } else if (onesided) {
                onesided_sender<BlockPolicy>(io, pp, vec_m1, kExtendLen);
            } else {
                sender<BlockPolicy>(io, pp, vec_m0, vec_m1, kExtendLen);
            }
        });
        std::thread receiver_worker([&]() {
            net::NetIO io("client", kAddress, kPort);
            if (optimized_sender) {
                result = reference_receiver(io, onesided);
            } else if (onesided) {
                result = onesided_receiver<BlockPolicy>(io, pp, vec_selection_bit, kExtendLen);
            } else {
                result = receiver<BlockPolicy>(io, pp, vec_selection_bit, kExtendLen);
            }
        });
        sender_worker.join();
        receiver_worker.join();

        std::vector<Block> expected;
        for (size_t i = 0; i < kExtendLen; ++i) {
            if (vec_selection_bit[i] == 1) {
                expected.push_back(vec_m1[i]);
            } else if (!onesided) {
                expected.push_back(vec_m0[i]);
            }
        }
        ASSERT_EQ(result.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(result[i], expected[i]) << "Selected index: " << i;
        }
    }

    int old_thread_num = 0;
};

TEST_P(ALSZOteBufferTest, Sender_Matches_Reference_Block_Receiver) {
    run_with_reference_peer(true, false);
}

TEST_P(ALSZOteBufferTest, Receiver_Matches_Reference_Block_Sender) {
    run_with_reference_peer(false, false);
}

TEST_P(ALSZOteBufferTest, Onesided_Sender_Matches_Reference_Block_Receiver) {
    run_with_reference_peer(true, true);
}

TEST_P(ALSZOteBufferTest, Onesided_Receiver_Matches_Reference_Block_Sender) {
    run_with_reference_peer(false, true);
}

// Preserve the previous cases, then check just below/at a window and beyond two.
INSTANTIATE_TEST_SUITE_P(BaseLengthThreadsChoicesAndRows, ALSZOteBufferTest,
    ::testing::Values(std::make_tuple(size_t{128}, 1, 0, size_t{16384 + 384}),
                      std::make_tuple(size_t{128}, 4, 1, size_t{16384 + 384}),
                      std::make_tuple(size_t{256}, 1, 2, size_t{16384 + 384}),
                      std::make_tuple(size_t{256}, 4, 2, size_t{16384 + 384}),
                      std::make_tuple(size_t{128}, 1, 3, size_t{32640}),
                      std::make_tuple(size_t{128}, 4, 3, size_t{32768}),
                      std::make_tuple(size_t{128}, 4, 3, size_t{65536 + 128})));

// ============================================================
// 6. Network Exceptions Escape the Parallel Region
// ============================================================

class ALSZOteDisconnectTest : public ALSZOteTest,
                             public ::testing::WithParamInterface<int> {
protected:
    void SetUp() override {
        ALSZOteTest::SetUp();
        old_thread_num = config::thread_num;
        config::thread_num = GetParam();
        kExtendLen = 32768 + 128;
    }

    void TearDown() override {
        config::thread_num = old_thread_num;
    }

    int old_thread_num = 0;
};

TEST_P(ALSZOteDisconnectTest, Sender_Reports_Peer_Disconnect_After_Base_OT) {
    std::exception_ptr sender_error, peer_error;
    bool base_ot_completed = false;
    std::thread sender_worker([&]() {
        try {
            net::NetIO io("server", kAddress, kPort);
            random_sender(io, pp, kExtendLen);
        } catch (...) {
            sender_error = std::current_exception();
        }
    });
    std::thread receiver_worker([&]() {
        try {
            net::NetIO io("client", kAddress, kPort);
            const Block salt = make_block(0x123456789abcdef0ULL, 0x0fedcba987654321ULL);
            auto seed = prg::set_seed(&salt, 0);
            auto t_seeds = prg::gen_random_blocks(seed, kBaseLen);
            auto u_seeds = prg::gen_random_blocks(seed, kBaseLen);
            mpc::np_ot::sender(io, pp.base_ot_pp, t_seeds, u_seeds, kBaseLen);
            base_ot_completed = true;
            // Close after the complete base OT transcript, without sending P.
        } catch (...) {
            peer_error = std::current_exception();
        }
    });
    sender_worker.join();
    receiver_worker.join();

    ASSERT_FALSE(peer_error);
    ASSERT_TRUE(base_ot_completed);
    ASSERT_TRUE(sender_error);
    EXPECT_THROW(std::rethrow_exception(sender_error), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(ThreadCounts, ALSZOteDisconnectTest, ::testing::Values(1, 4));

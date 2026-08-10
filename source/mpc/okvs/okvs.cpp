/****************************************************************************
 * @file      okvs.cpp
 * @brief     Oblivious Key-Value Store primitive.
 * @details   Implements the oblivious key-value store as described in
 *            "Blazing Fast PSI from Improved OKVS and Subfield VOLE":
 *            <https://eprint.iacr.org/2022/320>
 *            References the open-source implementation available at:
 *            <https://github.com/Visa-Research/volepsi.git>
 * @author    Yang Cao
 *****************************************************************************/

#include <taihang/mpc/okvs/okvs.hpp>
#include <taihang/mpc/okvs/baxos.hpp>
#include <taihang/common/config.hpp>
#include <taihang/common/check.hpp>
#include <taihang/crypto/prg.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace taihang::mpc::okvs {

BlockArrayValue::BlockArrayValue() : var() {}

BlockArrayValue BlockArrayValue::operator^(const BlockArrayValue& other) const {
    BlockArrayValue result;
    for (size_t i = 0; i < sizeof(var) / sizeof(Block); ++i) {
        result.var[i] = var[i] ^ other.var[i];
    }
    return result;
}

BlockArrayValue& BlockArrayValue::operator^=(const BlockArrayValue& other) {
    for (size_t i = 0; i < sizeof(var) / sizeof(Block); ++i) {
        var[i] ^= other.var[i];
    }
    return *this;
}

bool BlockArrayValue::operator!=(const BlockArrayValue& other) const {
    for (size_t i = 0; i < sizeof(var) / sizeof(Block); ++i) {
        if (!is_equal(var[i], other.var[i])) {
            return true;
        }
    }
    return false;
}

static inline uint64_t divide_mullhi_u64(uint64_t x, uint64_t y)
{
#if defined(_MSC_VER) && defined(LIBDIVIDE_X86_64)
    return __umulh(x, y);
#elif defined(__SIZEOF_INT128__)
    __uint128_t xl = x, yl = y;
    __uint128_t rl = xl * yl;
    return (uint64_t)(rl >> 64);
#else
    // full 128 bits are x0 * y0 + (x0 * y1 << 32) + (x1 * y0 << 32) + (x1 * y1 << 64)
    uint32_t mask = 0xFFFFFFFF;
    uint32_t x0 = (uint32_t)(x & mask);
    uint32_t x1 = (uint32_t)(x >> 32);
    uint32_t y0 = (uint32_t)(y & mask);
    uint32_t y1 = (uint32_t)(y >> 32);
    uint32_t x0y0_hi = static_cast<uint32_t>((static_cast<uint64_t>(x0) * y0) >> 32);
    uint64_t x0y1 = x0 * (uint64_t)y1;
    uint64_t x1y0 = x1 * (uint64_t)y0;
    uint64_t x1y1 = x1 * (uint64_t)y1;
    uint64_t temp = x1y0 + x0y0_hi;
    uint64_t temp_lo = temp & mask;
    uint64_t temp_hi = temp >> 32;

    return x1y1 + temp_hi + ((temp_lo + x0y1) >> 32);
#endif
}

__attribute__((target("avx2")))
static inline __m256i divide_mullhi_uint64_t_vec256(__m256i x, __m256i y)
{
    // see m128i variant for comments.
    __m256i x0y0 = _mm256_mul_epu32(x, y);
    __m256i x0y0_hi = _mm256_srli_epi64(x0y0, 32);

    __m256i x1 = _mm256_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 1, 1));
    __m256i y1 = _mm256_shuffle_epi32(y, _MM_SHUFFLE(3, 3, 1, 1));

    __m256i x0y1 = _mm256_mul_epu32(x, y1);
    __m256i x1y0 = _mm256_mul_epu32(x1, y);
    __m256i x1y1 = _mm256_mul_epu32(x1, y1);

    __m256i mask = _mm256_set1_epi64x(0xFFFFFFFF);
    __m256i temp = _mm256_add_epi64(x1y0, x0y0_hi);
    __m256i temp_lo = _mm256_and_si256(temp, mask);
    __m256i temp_hi = _mm256_srli_epi64(temp, 32);

    temp_lo = _mm256_srli_epi64(_mm256_add_epi64(temp_lo, x0y1), 32);
    temp_hi = _mm256_add_epi64(x1y1, temp_hi);
    return _mm256_add_epi64(temp_lo, temp_hi);
}

uint64_t divide_u64_do(uint64_t numer, const Divider *denom)
{
    uint8_t more = denom->more;
    if (!denom->magic)
    {
        // the divisor is a power of 2
        return numer >> more;
    }
    else
    {
        uint64_t q = divide_mullhi_u64(denom->magic, numer);
        // Determine whether it is an N+1 bit algorithm
        if (more & 0x40)
        {
            // N+1-bit algorithm
            uint64_t t = ((numer - q) >> 1) + q;
            return t >> (more & 0x3F);
        }
        else
        {
            // All upper bits are 0,
            // don't need to mask them off.
            return q >> more;
        }
    }
}

__attribute__((target("avx2")))
__m256i divide_u64_do_vec256(__m256i numers, const Divider *denom)
{
    uint8_t more = denom->more;
    if (!denom->magic)
    {
        return _mm256_srli_epi64(numers, more);
    }
    else
    {
        __m256i q = divide_mullhi_uint64_t_vec256(numers, _mm256_set1_epi64x(denom->magic));
        if (more & 0x40)
        {
            // uint32_t t = ((numer - q) >> 1) + q;
            // return t >> denom->shift;
            uint32_t shift = more & 0x3F;
            __m256i t = _mm256_add_epi64(_mm256_srli_epi64(_mm256_sub_epi64(numers, q), 1), q);
            return _mm256_srli_epi64(t, shift);
        }
        else
        {
            return _mm256_srli_epi64(q, more);
        }
    }
}
int32_t count_leading_zeros64(uint64_t val)
{
    return static_cast<int32_t>(std::countl_zero(val));
}

#if defined(__x86_64__) || defined(_M_X64)
#define LIBDIVIDE_X86_64
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LIBDIVIDE_GCC_STYLE_ASM
#endif


static inline uint64_t divide_128_div_64_to_64(
    uint64_t numhi, uint64_t numlo, uint64_t den, uint64_t *r) {
    // N.B. resist the temptation to use __uint128_t here.
    // In LLVM compiler-rt, it performs a 128/128 -> 128 division which is many times slower than
    // necessary. In gcc it's better but still slower than the divlu implementation, perhaps because
    // it's not LIBDIVIDE_INLINEd.
#if defined(LIBDIVIDE_X86_64) && defined(LIBDIVIDE_GCC_STYLE_ASM)
    uint64_t result;
    __asm__("divq %[v]" : "=a"(result), "=d"(*r) : [v] "r"(den), "a"(numlo), "d"(numhi));
    return result;
#else
    // We work in base 2**32.
    // A uint32 holds a single digit. A uint64 holds two digits.
    // Our numerator is conceptually [num3, num2, num1, num0].
    // Our denominator is [den1, den0].
    const uint64_t b = ((uint64_t)1 << 32);

    // The high and low digits of our computed quotient.
    uint32_t q1;
    uint32_t q0;

    // The normalization shift factor.
    int shift;

    // The high and low digits of our denominator (after normalizing).
    // Also the low 2 digits of our numerator (after normalizing).
    uint32_t den1;
    uint32_t den0;
    uint32_t num1;
    uint32_t num0;

    // A partial remainder.
    uint64_t rem;

    // The estimated quotient, and its corresponding remainder (unrelated to true remainder).
    uint64_t qhat;
    uint64_t rhat;

    // Variables used to correct the estimated quotient.
    uint64_t c1;
    uint64_t c2;

    // Check for overflow and divide by 0.
    if (numhi >= den) {
        if (r != NULL) *r = ~0ull;
        return ~0ull;
    }

    // Determine the normalization factor. We multiply den by this, so that its leading digit is at
    // least half b. In binary this means just shifting left by the number of leading zeros, so that
    // there's a 1 in the MSB.
    // We also shift numer by the same amount. This cannot overflow because numhi < den.
    // The expression (-shift & 63) is the same as (64 - shift), except it avoids the UB of shifting
    // by 64. The funny bitwise 'and' ensures that numlo does not get shifted into numhi if shift is
    // 0. clang 11 has an x86 codegen bug here: see LLVM bug 50118. The sequence below avoids it.
    shift = count_leading_zeros64(den);
    den <<= shift;
    numhi <<= shift;
    numhi |= (numlo >> (-shift & 63)) & (-(int64_t)shift >> 63);
    numlo <<= shift;

    // Extract the low digits of the numerator and both digits of the denominator.
    num1 = (uint32_t)(numlo >> 32);
    num0 = (uint32_t)(numlo & 0xFFFFFFFFu);
    den1 = (uint32_t)(den >> 32);
    den0 = (uint32_t)(den & 0xFFFFFFFFu);

    // We wish to compute q1 = [n3 n2 n1] / [d1 d0].
    // Estimate q1 as [n3 n2] / [d1], and then correct it.
    // Note while qhat may be 2 digits, q1 is always 1 digit.
    qhat = numhi / den1;
    rhat = numhi % den1;
    c1 = qhat * den0;
    c2 = rhat * b + num1;
    if (c1 > c2) qhat -= (c1 - c2 > den) ? 2 : 1;
    q1 = (uint32_t)qhat;

    // Compute the true (partial) remainder.
    rem = numhi * b + num1 - q1 * den;

    // We wish to compute q0 = [rem1 rem0 n0] / [d1 d0].
    // Estimate q0 as [rem1 rem0] / [d1] and correct it.
    qhat = rem / den1;
    rhat = rem % den1;
    c1 = qhat * den0;
    c2 = rhat * b + num0;
    if (c1 > c2) qhat -= (c1 - c2 > den) ? 2 : 1;
    q0 = (uint32_t)qhat;

    // Return remainder if requested.
    if (r != NULL) *r = (rem * b + num0 - q0 * den) >> shift;
    return ((uint64_t)q1 << 32) | q0;
#endif
}


Divider gen_divider(uint64_t d)
{
    if (d == 0)
    {
        throw std::invalid_argument("OKVS divider must be nonzero.");
    }

    Divider result;

    // The calculated effective number is 64-count_leading_zeros64(d) bits,
    // and the maximum number of digits represented by x bits is 2^x-1<2^x,
    // so floor(log2(d))=64-count_leading_zeros64(d)-1

    uint32_t floor_log_2_d = 63 - count_leading_zeros64(d);

    // Power of 2
    if ((d & (d - 1)) == 0)
    {
        result.magic = 0;
        result.more = (uint8_t)(floor_log_2_d);
        return result;
    }
    else
    {
        // Calculate using the round-up method
        // Reference: https://rubenvannieuwpoort.nl/posts/division-by-constant-unsigned-integers
        uint64_t proposed_m, rem;
        uint8_t more;
        // (1 << (64 + floor_log_2_d)) / d
        proposed_m = divide_128_div_64_to_64((uint64_t)1 << floor_log_2_d, 0, d, &rem);

        TAIHANG_ASSERT(rem > 0 && rem < d, "OKVS divider generation produced an invalid remainder.");

        // The distance of the next multiple of d from proposed_m
        const uint64_t e = d - rem;

        // This power works if e < 2**floor_log_2_d.
        // That means that m_up*d<=2^{N+l}+2^l,l=floor_log_2_d, The correctness holds by Theorem 3 or 10 in ref.
        if (e < ((uint64_t)1 << floor_log_2_d))
        {
            // This power works
            more = (uint8_t)floor_log_2_d;
        }
        else
        {
            // This means that the N(64)-bit round-up method is no longer applicable.
            // The N+1-bit round-up method follows Theorem 4; Theorem 10 also
            // permits a round-down variant.

            // In fact, what we want to calculate is proposed_m = (1<<(64+ceil(log2(d))))/d
            proposed_m += proposed_m;
            const uint64_t twice_rem = rem + rem;
            if (twice_rem >= d || twice_rem < rem)
                proposed_m += 1;

            // Determine which branch is entered based on whether it is ORed with 0x40
            more = (uint8_t)(floor_log_2_d | 0x40);
        }

        // calculate m_up(addition because of the ceil function)
        result.magic = 1 + proposed_m;
        // calculate l,which indicates the number of bits to shift
        // for N-bit algorithm,l=floor(log2(d)),
        // for N+1-bit algorithm,l=ceil(log2(d))-1=floor(log2(d)),
        result.more = more;
    }
    return result;
}

#define block_256 __m256i
__attribute__((target("avx2")))
void reduce_mod32(uint64_t *vals, Divider *div, const uint64_t &modVal)
{
    uint64_t i = 0;
    block_256 row256a = _mm256_loadu_si256((block_256 *)&vals[i]);
    block_256 row256b = _mm256_loadu_si256((block_256 *)&vals[i + 4]);
    block_256 row256c = _mm256_loadu_si256((block_256 *)&vals[i + 8]);
    block_256 row256d = _mm256_loadu_si256((block_256 *)&vals[i + 12]);
    block_256 row256e = _mm256_loadu_si256((block_256 *)&vals[i + 16]);
    block_256 row256f = _mm256_loadu_si256((block_256 *)&vals[i + 20]);
    block_256 row256g = _mm256_loadu_si256((block_256 *)&vals[i + 24]);
    block_256 row256h = _mm256_loadu_si256((block_256 *)&vals[i + 28]);

    auto tempa = divide_u64_do_vec256(row256a, div);
    auto tempb = divide_u64_do_vec256(row256b, div);
    auto tempc = divide_u64_do_vec256(row256c, div);
    auto tempd = divide_u64_do_vec256(row256d, div);
    auto tempe = divide_u64_do_vec256(row256e, div);
    auto tempf = divide_u64_do_vec256(row256f, div);
    auto tempg = divide_u64_do_vec256(row256g, div);
    auto temph = divide_u64_do_vec256(row256h, div);

    auto temp64a = (uint64_t *)&tempa;
    auto temp64b = (uint64_t *)&tempb;
    auto temp64c = (uint64_t *)&tempc;
    auto temp64d = (uint64_t *)&tempd;
    auto temp64e = (uint64_t *)&tempe;
    auto temp64f = (uint64_t *)&tempf;
    auto temp64g = (uint64_t *)&tempg;
    auto temp64h = (uint64_t *)&temph;

    vals[i + 0] -= temp64a[0] * modVal;
    vals[i + 1] -= temp64a[1] * modVal;
    vals[i + 2] -= temp64a[2] * modVal;
    vals[i + 3] -= temp64a[3] * modVal;
    vals[i + 4] -= temp64b[0] * modVal;
    vals[i + 5] -= temp64b[1] * modVal;
    vals[i + 6] -= temp64b[2] * modVal;
    vals[i + 7] -= temp64b[3] * modVal;
    vals[i + 8] -= temp64c[0] * modVal;
    vals[i + 9] -= temp64c[1] * modVal;
    vals[i + 10] -= temp64c[2] * modVal;
    vals[i + 11] -= temp64c[3] * modVal;
    vals[i + 12] -= temp64d[0] * modVal;
    vals[i + 13] -= temp64d[1] * modVal;
    vals[i + 14] -= temp64d[2] * modVal;
    vals[i + 15] -= temp64d[3] * modVal;
    vals[i + 16] -= temp64e[0] * modVal;
    vals[i + 17] -= temp64e[1] * modVal;
    vals[i + 18] -= temp64e[2] * modVal;
    vals[i + 19] -= temp64e[3] * modVal;
    vals[i + 20] -= temp64f[0] * modVal;
    vals[i + 21] -= temp64f[1] * modVal;
    vals[i + 22] -= temp64f[2] * modVal;
    vals[i + 23] -= temp64f[3] * modVal;
    vals[i + 24] -= temp64g[0] * modVal;
    vals[i + 25] -= temp64g[1] * modVal;
    vals[i + 26] -= temp64g[2] * modVal;
    vals[i + 27] -= temp64g[3] * modVal;
    vals[i + 28] -= temp64h[0] * modVal;
    vals[i + 29] -= temp64h[1] * modVal;
    vals[i + 30] -= temp64h[2] * modVal;
    vals[i + 31] -= temp64h[3] * modVal;
}
__attribute__((target("pclmul,sse2")))
Block gf128_mul(const Block x, const Block y)
{

    Block x0y0 = _mm_clmulepi64_si128(x, y, 0x00);
    Block x1y0 = _mm_clmulepi64_si128(x, y, 0x10);
    Block x0y1 = _mm_clmulepi64_si128(x, y, 0x01);
    Block x1y1 = _mm_clmulepi64_si128(x, y, 0x11);
    x1y0 = (x1y0 ^ x0y1);
    x0y1 = _mm_slli_si128(x1y0, 8);
    x1y0 = _mm_srli_si128(x1y0, 8);
    x0y0 = (x0y0 ^ x0y1);
    x1y1 = (x1y1 ^ x1y0);

    auto mul256_low = x0y0;
    auto mul256_high = x1y1;

    static const constexpr std::uint64_t mod_omit128 = 0b10000111;

    const Block modulus_omit128 = _mm_loadl_epi64(reinterpret_cast<const __m128i_u*>(&mod_omit128));
    Block impact = _mm_clmulepi64_si128(mul256_high, modulus_omit128, 0x01);
    mul256_low = _mm_xor_si128(mul256_low, _mm_slli_si128(impact, 8));
    mul256_high = _mm_xor_si128(mul256_high, _mm_srli_si128(impact, 8));

    impact = _mm_clmulepi64_si128(mul256_high, modulus_omit128, 0x00);
    mul256_low = _mm_xor_si128(mul256_low, impact);

    return mul256_low;
}

// gf128_mul overload and return z : z.var[i] = gf128_mul(x.var[i], y)
BlockArrayValue gf128_mul(const BlockArrayValue x, const Block y)
{
    BlockArrayValue result;
    for (size_t i = 0; i < sizeof(result.var) / sizeof(Block); ++i) {
        result.var[i] = gf128_mul(x.var[i], y);
    }
    return result;
}



Block gf128_inv(const Block x)
{
    // x^{-1}=x^{2^128-1-1}=x^{2^128-2}=x^{0b1111...0} the count of 1 in x = 127 ,the count of 0 in x = 1
    Block a = x;

    Block result = kZeroBlock;
    for (uint8_t i = 0; i <= 6; ++i)
    {
        /* entering the loop a = x^{0b1...1}  the count of 1 in a = 2^i  */
        Block b = a;
        for (uint8_t j = 0; j < (1 << i); ++j)
        {
            b = gf128_mul(b, b);
        }
        /* after the loop b = a^{2^{2^i}} = x^{0b1...10...0} the count of 0 in b = 2^i  */
        a = gf128_mul(a, b);
        /* now a = x^{0b1...1} the count of 1 in a = 2^{i+1}*/

        if (i == 0)
        {
            result = b;
        }
        else
        {
            result = gf128_mul(result, b);
            /* now result = x^{0b1...10} the count of 1 in result = 2^{i+1}-1 */
        }
    }

    /* finally result = x^{0b1...10} the count of 1 in result = 2^7-1 =127. That's what we need.*/
    // auto one_block = make_block(0, 1);
    // auto mul_result = gf128_mul(result, x);
    /* now result = x^{2^128-2} */
    return result;
}
bool prev_combination(std::vector<uint8_t>& comb, uint64_t n) {
    int k = comb.size();
    for (int i = k - 1; i >= 0; --i) {
        if (comb[i] < n - k + i) {
            ++comb[i];
            for (int j = i + 1; j < k; ++j) {
                comb[j] = comb[j - 1] + 1;
            }
            return true;
        }
    }
    return false;
}

bool check_invert_gf128(std::vector<std::vector<Block>> &mat)
{
    const size_t n = mat.size();
    std::vector<std::vector<Block>> Inv(n, std::vector<Block>(n, kZeroBlock));
    auto one_block = make_block(0, 1);
    for (size_t i = 0; i < n; i++)
        Inv[i][i] = one_block;

    for (size_t i = 0; i < n; i++)
    {
        if (is_equal(mat[i][i], kZeroBlock))
        {
            for (size_t j = i + 1; j < n; j++)
            {
                if (is_equal(mat[j][i], one_block))
                {
                    mat[i].swap(mat[j]);
                    Inv[i].swap(Inv[j]);
                    break;
                }
            }

            if (is_equal(mat[i][i], kZeroBlock))
                return false;
        }

        auto mat_i_i_inv = gf128_inv(mat[i][i]);

        for (size_t j = 0; j < n; j++)
        {
            if (i == j)
            {
                mat[i][j] = one_block;
            }
            else
            {
                mat[i][j] = gf128_mul(mat[i][j], mat_i_i_inv);
            }
            Inv[i][j] = gf128_mul(Inv[i][j], mat_i_i_inv);
        }

        for (size_t j = 0; j < n; j++)
        {
            if (j != i)
            {
                auto mat_j_i = mat[j][i];
                for (size_t k = 0; k < n; k++)
                {
                    mat[j][k] ^= gf128_mul(mat[i][k], mat_j_i);
                    Inv[j][k] ^= gf128_mul(Inv[i][k], mat_j_i);
                }
            }
        }
    }

    mat = Inv;
    return true;
}

uint64_t col_to_dec(std::vector<uint64_t> &binary)
{
    int decimal = 0;
    for (int i = binary.size() - 1; i >= 0; i--)
    {
        decimal = decimal << 1 | binary[i];
    }
    return decimal;
}

bool check_invert(std::vector<std::vector<uint8_t>> &mat)
{
    const size_t n = mat.size();
    std::vector<std::vector<uint8_t>> Inv(n, std::vector<uint8_t>(n, 0));
    for (size_t i = 0; i < n; i++)
        Inv[i][i] = 1;

    for (size_t i = 0; i < n; i++)
    {
        if (mat[i][i] == 0)
        {

            for (size_t j = i + 1; j < n; j++)
            {
                if (mat[j][i] == 1)
                {
                    mat[i].swap(mat[j]);
                    Inv[i].swap(Inv[j]);
                    break;
                }
            }

            if (mat[i][i] == 0)
                return false;
        }

        for (size_t j = 0; j < n; j++)
        {
            if (j != i && mat[j][i])
            {
                for (size_t k = 0; k < n; k++)
                {
                    mat[j][k] ^= mat[i][k];
                    Inv[j][k] ^= Inv[i][k];
                }
            }
        }
    }

    mat = Inv;
    return true;
}
uint64_t log2_floor(uint64_t x)
{
    return 64 - count_leading_zeros64(x) - 1;
}
uint64_t log2_ceil(uint64_t x)
{
    return 64 - count_leading_zeros64(x - 1);
}

const std::vector<std::vector<double>> sizes{{
    // log #bins        , log #balls
    //                    0, 1, 2, 3, 4, 5, 6,...
    /*0*/ {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}},
    /*1*/ {{0, 1, 2, 3, 4, 5, 5.857980995, 6.686500527, 7.523561956, 8.392317423, 9.290018847, 10.21067134, 11.15228484, 12.10950422, 13.07831762, 14.05579081, 15.03969019, 16.02818667, 17.01999243, 18.01416217, 19.01002812, 20.00709846, 21.00502277, 22.00355291, 23.00251306, 24.00177721, 25.00125665, 26.00088843, 27.00062802, 28.00044386, 29.00031363}},
    /*2*/ {{0, 1, 2, 3, 4, 4.754887502, 5.459431619, 6.129283017, 6.87036472, 7.665335917, 8.491853096, 9.361943774, 10.2632692, 11.18982456, 12.13602985, 13.09737377, 14.06936585, 15.04929552, 16.03499235, 17.02481502, 18.01757508, 19.01244722, 20.00880883, 21.00623293, 22.00440908, 23.00311846, 24.00220528, 25.00155925, 26.00110233, 27.00077917, 28.00055063}},
    /*3*/ {{0, 1, 2, 3, 3.906890596, 4.459431619, 5, 5.614709844, 6.247927513, 6.965784285, 7.727920455, 8.539158811, 9.394462695, 10.28771238, 11.20762447, 12.14879432, 13.10639936, 14.07581338, 15.05392588, 16.03827601, 17.02713927, 18.01923236, 19.0136171, 20.00963729, 21.00681914, 22.00482395, 23.00341171, 24.00241262, 25.00170588, 26.00120598, 27.00085241}},
    /*4*/ {{0, 1, 2, 3, 3.700439718, 4.087462841, 4.584962501, 5.129283017, 5.672425342, 6.321928095, 7.011227255, 7.761551232, 8.566054038, 9.413627929, 10.30149619, 11.21735191, 12.15608308, 13.11146174, 14.07948478, 15.05651071, 16.04011845, 17.0284457, 18.02015526, 19.01427116, 20.0101019, 21.00714706, 22.00505602, 23.0035759, 24.00252877, 25.00178798, 26.00126401}},
    /*5*/ {{0, 1, 2, 3, 3.459431619, 3.807354922, 4.247927513, 4.700439718, 5.169925001, 5.727920455, 6.357552005, 7.033423002, 7.781359714, 8.581200582, 9.426264755, 10.30947635, 11.22339841, 12.16050175, 13.11471839, 14.08173308, 15.06149777, 16.04127413, 17.02926566, 18.02074126, 19.01468524, 20.01039425, 21.00735446, 22.00520271, 23.00367969, 24.00260216, 25.0018399}},
    /*6*/ {{0, 1, 2, 2.807354922, 3.169925001, 3.584962501, 3.906890596, 4.321928095, 4.700439718, 5.209453366, 5.754887502, 6.375039431, 7.055282436, 7.794415866, 8.592457037, 9.434628228, 10.31514956, 11.22821744, 12.16364968, 13.11699368, 14.08339622, 15.05930221, 16.04210821, 17.02985876, 18.0211535, 19.01497939, 20.01060323, 21.00750229, 22.00530724, 23.00375379, 24.00265452}},
    /*7*/ {{0, 1, 2, 2.807354922, 3, 3.321928095, 3.584962501, 4, 4.321928095, 4.754887502, 5.247927513, 5.781359714, 6.392317423, 7.06608919, 7.807354922, 8.599912842, 9.440869168, 10.32080055, 11.23122118, 12.16616308, 13.11877889, 14.08472536, 15.06023151, 16.04277086, 17.03032228, 18.02148428, 19.0152163, 20.01076984, 21.00762, 22.00539086, 23.0038128}},
    /*8*/ {{0, 1, 2, 2.584962501, 2.807354922, 3.169925001, 3.321928095, 3.700439718, 4, 4.392317423, 4.807354922, 5.247927513, 5.807354922, 6.409390936, 7.076815597, 7.813781191, 8.607330314, 9.445014846, 10.32418055, 11.23421868, 12.16835873, 13.1203999, 14.08580438, 15.0610336, 16.04332639, 17.03073179, 18.02177162, 19.01541777, 20.01091459, 21.00772264, 22.00546316}},
    /*9*/ {{0, 1, 2, 2.321928095, 2.584962501, 3, 3.169925001, 3.459431619, 3.700439718, 4, 4.392317423, 4.807354922, 5.285402219, 5.807354922, 6.426264755, 7.087462841, 7.826548487, 8.614709844, 9.451211112, 10.32755264, 11.23720996, 12.17023805, 13.12185725, 14.08679969, 15.06175089, 16.04386035, 17.03109809, 18.02203723, 19.01560561, 20.01104704, 21.00781707}},
    /*10*/ {{0, 1, 2, 2.321928095, 2.584962501, 2.807354922, 3, 3.169925001, 3.459431619, 3.700439718, 4.087462841, 4.392317423, 4.807354922, 5.285402219, 5.832890014, 6.426264755, 7.098032083, 7.832890014, 8.618385502, 9.45532722, 10.33203655, 11.23959853, 12.17211493, 13.12315143, 14.0877943, 15.06246782, 16.04435143, 17.03145353, 18.02229195, 19.01578526, 20.01117265}},
    /*11*/ {{0, 1, 2, 2.321928095, 2.321928095, 2.584962501, 2.807354922, 3, 3.169925001, 3.459431619, 3.700439718, 4.087462841, 4.459431619, 4.857980995, 5.321928095, 5.832890014, 6.442943496, 7.108524457, 7.839203788, 8.625708843, 9.459431619, 10.33539035, 11.24198315, 12.17398937, 13.12444446, 14.08878824, 15.06314225, 16.04484233, 17.03179811, 18.02253579, 19.01595672}},
    /*12*/ {{0, 1, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.807354922, 3, 3.169925001, 3.459431619, 3.807354922, 4.087462841, 4.459431619, 4.857980995, 5.321928095, 5.857980995, 6.459431619, 7.118941073, 7.845490051, 8.62935662, 9.463524373, 10.33873638, 11.24436384, 12.17586138, 13.12573633, 14.08969874, 15.06381637, 16.04531174, 17.03213185, 18.02277416}},
    /*13*/ {{0, 1, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 3, 3.321928095, 3.459431619, 3.807354922, 4.087462841, 4.459431619, 4.857980995, 5.357552005, 5.882643049, 6.459431619, 7.129283017, 7.851749041, 8.636624621, 9.46760555, 10.34207467, 11.2467406, 12.17741954, 13.12702704, 14.09060867, 15.06444807, 16.04575966, 17.0324655}},
    /*14*/ {{0, 1, 1.584962501, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 3, 3.321928095, 3.584962501, 3.807354922, 4.169925001, 4.459431619, 4.906890596, 5.357552005, 5.882643049, 6.475733431, 7.139551352, 7.857980995, 8.64385619, 9.47370575, 10.34429591, 11.24911345, 12.1792871, 13.1283166, 14.09151803, 15.06507949, 16.04622877}},
    /*15*/ {{0, 1, 1.584962501, 2, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.807354922, 2.807354922, 3.169925001, 3.321928095, 3.584962501, 3.807354922, 4.169925001, 4.523561956, 4.906890596, 5.357552005, 5.906890596, 6.491853096, 7.14974712, 7.87036472, 8.647458426, 9.477758266, 10.34762137, 11.25148241, 12.18084157, 13.12944402, 14.09234422, 15.06571064}},
    /*16*/ {{0, 1, 1.584962501, 1.584962501, 2, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.807354922, 3, 3.169925001, 3.321928095, 3.584962501, 3.906890596, 4.169925001, 4.523561956, 4.906890596, 5.392317423, 5.906890596, 6.491853096, 7.159871337, 7.876516947, 8.654636029, 9.481799432, 10.35093918, 11.25384748, 12.18270471, 13.13073143, 14.09325249}},
    /*17*/ {{0, 1, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.807354922, 3, 3.169925001, 3.321928095, 3.584962501, 3.906890596, 4.169925001, 4.523561956, 4.95419631, 5.392317423, 5.930737338, 6.50779464, 7.159871337, 7.882643049, 8.658211483, 9.485829309, 10.35424938, 11.25620869, 12.1842555, 13.13185696}},
    /*18*/ {{0, 1, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 3, 3.169925001, 3.321928095, 3.584962501, 3.906890596, 4.169925001, 4.584962501, 4.95419631, 5.426264755, 5.930737338, 6.523561956, 7.169925001, 7.888743249, 8.665335917, 9.48984796, 10.357552, 11.25856603, 12.18580462}},
    /*19*/ {{0, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 3, 3.169925001, 3.459431619, 3.700439718, 3.906890596, 4.247927513, 4.584962501, 4.95419631, 5.426264755, 5.95419631, 6.523561956, 7.17990909, 7.894817763, 8.668884984, 9.493855449, 10.35974956, 11.26091953}},
    /*20*/ {{0, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 3, 3.169925001, 3.459431619, 3.700439718, 3.906890596, 4.247927513, 4.584962501, 5, 5.426264755, 5.95419631, 6.539158811, 7.189824559, 7.900866808, 8.675957033, 9.497851837, 10.36303963}},
    /*21*/ {{0, 1, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 3, 3.169925001, 3.459431619, 3.700439718, 4, 4.247927513, 4.584962501, 5, 5.459431619, 5.977279923, 6.554588852, 7.199672345, 7.906890596, 8.6794801, 9.501837185}},
    /*22*/ {{0, 1, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2.321928095, 2.321928095, 2.321928095, 2.584962501, 2.807354922, 2.807354922, 3, 3.321928095, 3.459431619, 3.700439718, 4, 4.247927513, 4.64385619, 5, 5.459431619, 5.977279923, 6.554588852, 7.209453366, 7.912889336, 8.682994584}},
    /*23*/ {{0, 1, 1, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2.321928095, 2.321928095, 2.321928095, 2.584962501, 2.807354922, 2.807354922, 3, 3.321928095, 3.459431619, 3.700439718, 4, 4.321928095, 4.64385619, 5.044394119, 5.491853096, 6, 6.569855608, 7.209453366, 7.918863237}},
    /*24*/ {{0, 1, 1, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 3, 3.169925001, 3.321928095, 3.459431619, 3.700439718, 4, 4.321928095, 4.64385619, 5.044394119, 5.491853096, 6, 6.584962501, 7.21916852}},
    /*25*/ {{0, 1, 1, 1, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 3, 3.169925001, 3.321928095, 3.459431619, 3.807354922, 4, 4.321928095, 4.64385619, 5.044394119, 5.491853096, 6.022367813, 6.584962501}},
    /*26*/ {{0, 1, 1, 1, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 3, 3.169925001, 3.321928095, 3.584962501, 3.807354922, 4, 4.321928095, 4.700439718, 5.087462841, 5.523561956, 6.022367813}},
    /*27*/ {{0, 1, 1, 1, 1, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2.321928095, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 3, 3.169925001, 3.321928095, 3.584962501, 3.807354922, 4.087462841, 4.392317423, 4.700439718, 5.087462841, 5.523561956}},
    /*28*/ {{0, 1, 1, 1, 1, 1, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2.321928095, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 3, 3.169925001, 3.321928095, 3.584962501, 3.807354922, 4.087462841, 4.392317423, 4.700439718, 5.087462841}},
    /*29*/ {{0, 1, 1, 1, 1, 1, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2, 2.321928095, 2.321928095, 2.321928095, 2.584962501, 2.807354922, 2.807354922, 3, 3.169925001, 3.321928095, 3.584962501, 3.807354922, 4.087462841, 4.392317423, 4.754887502}},
    /*30*/ {{0, 1, 1, 1, 1, 1, 1, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 2.807354922, 3, 3.169925001, 3.459431619, 3.584962501, 3.807354922, 4.087462841, 4.392317423}},
    /*31*/ {{0, 1, 1, 1, 1, 1, 1, 1, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 2.807354922, 3, 3.169925001, 3.459431619, 3.584962501, 3.906890596, 4.087462841}},
    /*32*/ {{0, 1, 1, 1, 1, 1, 1, 1, 1, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 1.584962501, 2, 2, 2, 2.321928095, 2.321928095, 2.321928095, 2.584962501, 2.584962501, 2.807354922, 3, 3, 3.321928095, 3.459431619, 3.700439718, 3.906890596}},
}};

uint64_t hashtable_bin_size(uint64_t bin_num, uint64_t item_num, uint8_t lambda)
{
    if (bin_num < 2)
        return item_num;

    const size_t bin_num_low = 64 - count_leading_zeros64(bin_num) - 1;
    const size_t bin_num_high = 64 - count_leading_zeros64(bin_num - 1);
    const size_t item_num_low = 64 - count_leading_zeros64(item_num) - 1;
    const size_t item_num_high = 64 - count_leading_zeros64(item_num - 1);

    auto bin_diff = std::log2(bin_num) - bin_num_low;
    auto item_diff = std::log2(item_num) - item_num_low;

    if (bin_num_high < sizes.size() && item_num_high < sizes[bin_num_high].size())
    {
        auto a0 = (bin_diff)*sizes[bin_num_low][item_num_low] + (1 - bin_diff) * sizes[bin_num_high][item_num_low];
        auto a1 = (bin_diff)*sizes[bin_num_low][item_num_high] + (1 - bin_diff) * sizes[bin_num_high][item_num_high];

        auto b0 = (item_diff)*a0 + (1 - item_diff) * a1;

        auto B = std::ceil(std::pow(2, b0));
        return B;
    }
    return 0;
}



uint8_t checked_byte(size_t value, const char* name) {
    if (value > std::numeric_limits<uint8_t>::max()) {
        throw std::invalid_argument(std::string("OKVS ") + name + " exceeds uint8_t.");
    }
    return static_cast<uint8_t>(value);
}

uint8_t thread_num() {
    const int configured = std::max(1, config::thread_num);
    return static_cast<uint8_t>(std::min(configured, static_cast<int>(std::numeric_limits<uint8_t>::max())));
}

struct BlockLess {
    bool operator()(const Block& lhs, const Block& rhs) const {
        return is_less_than(lhs, rhs);
    }
};

std::vector<Block> pad_keys(const PublicParameters& pp,
                            const std::vector<Block>& keys) {
    std::vector<Block> padded = keys;
    padded.reserve(pp.item_num);

    std::set<Block, BlockLess> used(keys.begin(), keys.end());
    auto seed = prg::set_seed(&pp.seed, 1);
    while (padded.size() < pp.item_num) {
        const Block candidate = prg::gen_random_blocks(seed, 1)[0];
        if (used.insert(candidate).second) {
            padded.push_back(candidate);
        }
    }
    return padded;
}

std::vector<Block> pad_values(const std::vector<Block>& values, size_t target_size) {
    std::vector<Block> padded = values;
    padded.resize(target_size, kZeroBlock);
    return padded;
}

template <DenseType dense_type>
void fill_derived_parameters(PublicParameters& pp) {
    // Baxos is a multi-thread clustered OKVS and is generally used instead of OKVS.
    auto okvs_seed = prg::set_seed(&pp.seed, 0);
    Baxos<dense_type, Block> baxos(pp.item_num,
                                   pp.bin_size,
                                   pp.sparse_weight,
                                   checked_byte(pp.statistical_security_parameter, "statistical security parameter"),
                                   &okvs_seed);
    pp.bin_num = baxos.bin_num;
    pp.item_num_per_bin = baxos.item_num_per_bin;
    pp.sparse_size = baxos.sparse_size;
    pp.dense_size = baxos.dense_size;
    pp.total_size = baxos.total_size;
    pp.storage_size = baxos.bin_num * baxos.total_size;
}

void validate_public_parameters(const PublicParameters& pp) {
    if (pp.item_num == 0) {
        throw std::invalid_argument("OKVS item_num must be positive.");
    }
    if (pp.bin_size == 0) {
        throw std::invalid_argument("OKVS bin_size must be positive.");
    }
    if (pp.sparse_weight < 2) {
        throw std::invalid_argument("OKVS sparse_weight must be at least 2.");
    }
    checked_byte(pp.statistical_security_parameter, "statistical security parameter");
}

template <DenseType dense_type>
Baxos<dense_type, Block> make_baxos(const PublicParameters& pp) {
    auto okvs_seed = prg::set_seed(&pp.seed, 0);
    return Baxos<dense_type, Block>(pp.item_num,
                                    pp.bin_size,
                                    pp.sparse_weight,
                                    checked_byte(pp.statistical_security_parameter, "statistical security parameter"),
                                    &okvs_seed);
}

template <DenseType dense_type>
std::vector<Block> encode_impl(const PublicParameters& pp,
                               const std::vector<Block>& keys,
                               const std::vector<Block>& values,
                               prg::Seed* prng) {
    auto baxos = make_baxos<dense_type>(pp);
    std::vector<Block> encoded(pp.storage_size);

    if (keys.size() == pp.item_num) {
        baxos.solve(keys, values, encoded, prng, thread_num());
        return encoded;
    }

    auto padded_keys = pad_keys(pp, keys);
    auto padded_values = pad_values(values, padded_keys.size());
    // Solve/encode the given input/value pair. The Paxos data structure is written to output.
    baxos.solve(padded_keys, padded_values, encoded, prng, thread_num());
    return encoded;
}

template <DenseType dense_type>
std::vector<Block> decode_impl(const PublicParameters& pp,
                               const std::vector<Block>& keys,
                               const std::vector<Block>& encoded) {
    auto baxos = make_baxos<dense_type>(pp);
    std::vector<Block> values(keys.size());
    // Decode the given input vector and write the result to values.
    baxos.decode(keys, values, encoded, thread_num());
    return values;
}

std::string PublicParameters::format() const {
    const char* dense_type_label = "Unknown";
    switch (dense_type) {
    case DenseType::Binary:
        dense_type_label = "Binary";
        break;
    case DenseType::Gf128:
        dense_type_label = "Gf128";
        break;
    }

    std::ostringstream oss;
    oss << "[OKVS PublicParameters]\n";
    oss << "Item number                    :" << item_num << "\n";
    oss << "Requested bin size             :" << bin_size << "\n";
    oss << "Bin number                     :" << bin_num << "\n";
    oss << "Item number per bin            :" << item_num_per_bin << "\n";
    oss << "Sparse weight                  :" << static_cast<int>(sparse_weight) << "\n";
    oss << "Statistical security parameter :" << statistical_security_parameter << "\n";
    oss << "Dense type                     :" << dense_type_label << "\n";
    oss << "Sparse size per bin            :" << sparse_size << "\n";
    oss << "Dense size per bin             :" << dense_size << "\n";
    oss << "Total size per bin             :" << total_size << "\n";
    oss << "Encoded storage size           :" << storage_size << "\n";
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const PublicParameters& pp) {
    uint64_t seed_words[2]{};
    const auto seed_bytes = to_bytes(pp.seed);
    std::memcpy(seed_words, seed_bytes.data(), seed_bytes.size());

    os << pp.item_num << " "
       << pp.bin_size << " "
       << static_cast<int>(pp.sparse_weight) << " "
       << pp.statistical_security_parameter << " "
       << static_cast<int>(pp.dense_type) << " "
       << seed_words[0] << " "
       << seed_words[1];
    return os;
}

std::istream& operator>>(std::istream& is, PublicParameters& pp) {
    int sparse_weight = 0;
    int dense_type = 0;
    uint64_t seed_first = 0;
    uint64_t seed_second = 0;

    size_t item_num = 0;
    size_t bin_size = 0;
    size_t statistical_security_parameter = 0;
    is >> item_num
       >> bin_size
       >> sparse_weight
       >> statistical_security_parameter
       >> dense_type
       >> seed_first
       >> seed_second;

    if (is) {
        pp = setup(item_num,
                   bin_size,
                   static_cast<uint8_t>(sparse_weight),
                   statistical_security_parameter,
                   static_cast<DenseType>(dense_type),
                   make_block(seed_second, seed_first));
    }
    return is;
}

PublicParameters setup(size_t item_num,
                       size_t bin_size,
                       uint8_t sparse_weight,
                       size_t statistical_security_parameter,
                       DenseType dense_type,
                       const Block& seed) {
    PublicParameters pp;
    pp.item_num = item_num;
    pp.bin_size = bin_size;
    pp.sparse_weight = sparse_weight;
    pp.statistical_security_parameter = statistical_security_parameter;
    pp.dense_type = dense_type;
    pp.seed = seed;

    validate_public_parameters(pp);

    switch (dense_type) {
    case DenseType::Binary:
        fill_derived_parameters<DenseType::Binary>(pp);
        break;
    case DenseType::Gf128:
        fill_derived_parameters<DenseType::Gf128>(pp);
        break;
    default:
        throw std::invalid_argument("Unknown OKVS dense type.");
    }

    return pp;
}

std::vector<Block> encode(const PublicParameters& pp,
                          const std::vector<Block>& keys,
                          const std::vector<Block>& values,
                          prg::Seed* prng) {
    validate_public_parameters(pp);
    if (keys.size() > pp.item_num || values.size() != keys.size()) {
        throw std::invalid_argument("OKVS encode input size mismatch.");
    }

    switch (pp.dense_type) {
    case DenseType::Binary:
        return encode_impl<DenseType::Binary>(pp, keys, values, prng);
    case DenseType::Gf128:
        return encode_impl<DenseType::Gf128>(pp, keys, values, prng);
    default:
        throw std::invalid_argument("Unknown OKVS dense type.");
    }
}

std::vector<Block> decode(const PublicParameters& pp,
                          const std::vector<Block>& keys,
                          const std::vector<Block>& encoded) {
    validate_public_parameters(pp);
    if (encoded.size() != pp.storage_size) {
        throw std::invalid_argument("OKVS encoded storage size mismatch.");
    }
    if (keys.empty()) {
        return {};
    }

    switch (pp.dense_type) {
    case DenseType::Binary:
        return decode_impl<DenseType::Binary>(pp, keys, encoded);
    case DenseType::Gf128:
        return decode_impl<DenseType::Gf128>(pp, keys, encoded);
    default:
        throw std::invalid_argument("Unknown OKVS dense type.");
    }
}

} // namespace taihang::mpc::okvs

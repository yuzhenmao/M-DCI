/*
 * BF16 (bfloat16) Utility Functions - Implementation
 *
 * This file implements conversion functions between FP32 and BF16 formats,
 * leveraging Intel AVX512_BF16 instructions when available.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "bf16_util.h"
#include <string.h>

// Include SIMD intrinsics if available
#if defined(__AVX512BF16__)
#include <immintrin.h>
#endif

// ============================================================================
// CPU Feature Detection
// ============================================================================

int cpu_has_avx512_bf16(void) {
#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang builtin for runtime CPU feature detection
    // Requires GCC 11+ or Clang 12+
    return __builtin_cpu_supports("avx512bf16");
#elif defined(_MSC_VER)
    // MSVC: Would need __cpuidex, but most Linux systems use GCC/Clang
    return 0;
#else
    // Conservative fallback
    return 0;
#endif
}

// ============================================================================
// Batch Conversion Functions
// ============================================================================

void f32_to_bf16_array(const float* src, bf16_t* dst, int n) {
#if defined(__AVX512BF16__)
    // Use hardware AVX512_BF16 instructions for optimal performance
    int i = 0;

    // Process 16 floats at a time (one AVX-512 register)
    for (; i + 15 < n; i += 16) {
        // Load 16 FP32 values
        __m512 fp32_vec = _mm512_loadu_ps(&src[i]);

        // Convert to BF16 using hardware instruction
        // _mm512_cvtneps_pbh converts 16 FP32 → 16 BF16 with RNE rounding
        __m256bh bf16_vec = _mm512_cvtneps_pbh(fp32_vec);

        // Store 16 BF16 values (256 bits = 16 * 16-bit)
        _mm256_storeu_epi16((__m256i*)(&dst[i]), (__m256i)bf16_vec);
    }

    // Handle remaining elements (less than 16)
    for (; i < n; i++) {
        dst[i] = f32_to_bf16(src[i]);
    }

#else
    // Software fallback: use scalar conversion
    for (int i = 0; i < n; i++) {
        dst[i] = f32_to_bf16(src[i]);
    }
#endif
}

void bf16_to_f32_array(const bf16_t* src, float* dst, int n) {
#if defined(__AVX512BF16__)
    // Use hardware AVX512_BF16 instructions
    int i = 0;

    // Process 16 BF16 values at a time
    for (; i + 15 < n; i += 16) {
        // Load 16 BF16 values (256 bits)
        __m256bh bf16_vec = (__m256bh)_mm256_loadu_epi16((__m256i const*)(&src[i]));

        // Convert to FP32 using hardware instruction
        // _mm512_cvtpbh_ps converts 16 BF16 → 16 FP32 (lossless)
        __m512 fp32_vec = _mm512_cvtpbh_ps(bf16_vec);

        // Store 16 FP32 values
        _mm512_storeu_ps(&dst[i], fp32_vec);
    }

    // Handle remaining elements
    for (; i < n; i++) {
        dst[i] = bf16_to_f32(src[i]);
    }

#else
    // Software fallback: use scalar conversion
    for (int i = 0; i < n; i++) {
        dst[i] = bf16_to_f32(src[i]);
    }
#endif
}

// ============================================================================
// Additional Utility Functions (for future use)
// ============================================================================

/**
 * Print BF16 value in binary format (for debugging)
 */
void bf16_print_binary(bf16_t x) {
    printf("BF16: ");
    for (int i = 15; i >= 0; i--) {
        printf("%d", (x >> i) & 1);
        if (i == 15 || i == 7) printf(" ");  // Separate sign, exponent, mantissa
    }
    printf(" (%.6f)\n", bf16_to_f32(x));
}

/**
 * Compare two BF16 arrays for equality (for testing)
 * Returns 1 if all elements are equal, 0 otherwise
 */
int bf16_array_equal(const bf16_t* a, const bf16_t* b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}

/**
 * Compute maximum absolute difference between BF16 and FP32 arrays (for testing)
 * Useful for validating conversion accuracy
 */
float bf16_max_abs_diff(const bf16_t* bf16_arr, const float* fp32_arr, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float bf16_as_fp32 = bf16_to_f32(bf16_arr[i]);
        float diff = bf16_as_fp32 - fp32_arr[i];
        if (diff < 0) diff = -diff;  // abs
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

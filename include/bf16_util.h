/*
 * BF16 (bfloat16) Utility Functions for DCI
 *
 * This file provides type definitions, conversion functions, and utilities
 * for working with bfloat16 (BF16) format using Intel AVX512_BF16 instructions.
 *
 * BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
 * - Same exponent range as FP32 (better for ML than FP16)
 * - Reduced precision (7 bits vs 23 bits mantissa)
 * - 50% memory savings vs FP32
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef BF16_UTIL_H
#define BF16_UTIL_H

#include <stdint.h>
#include <stdbool.h>

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * BF16 storage type - stored as 16-bit unsigned integer
 * Layout: [sign(1)] [exponent(8)] [mantissa(7)]
 */
typedef uint16_t bf16_t;

// ============================================================================
// CPU Feature Detection
// ============================================================================

/**
 * Check if CPU supports AVX512_BF16 instructions at runtime
 *
 * @return 1 if AVX512_BF16 is supported, 0 otherwise
 */
int cpu_has_avx512_bf16(void);

// ============================================================================
// Scalar Conversion Functions
// ============================================================================

/**
 * Convert single FP32 value to BF16 with round-to-nearest-even
 *
 * Uses hardware instruction if available, otherwise software emulation
 * with proper rounding.
 *
 * @param x Input FP32 value
 * @return BF16 representation
 */
static inline bf16_t f32_to_bf16(float x);

/**
 * Convert single BF16 value to FP32
 *
 * This is a lossless conversion (padding mantissa with zeros)
 *
 * @param x Input BF16 value
 * @return FP32 representation
 */
static inline float bf16_to_f32(bf16_t x);

// ============================================================================
// Batch Conversion Functions (Optimized with SIMD)
// ============================================================================

/**
 * Convert array of FP32 values to BF16
 *
 * Uses AVX512_BF16 intrinsics (_mm512_cvtneps_pbh) when available for
 * optimal performance. Processes 16 elements at a time with hardware support.
 *
 * @param src Source array of FP32 values
 * @param dst Destination array for BF16 values (must be pre-allocated)
 * @param n Number of elements to convert
 */
void f32_to_bf16_array(const float* src, bf16_t* dst, int n);

/**
 * Convert array of BF16 values to FP32
 *
 * Uses AVX512_BF16 intrinsics (_mm512_cvtpbh_ps) when available.
 * Processes 16 elements at a time with hardware support.
 *
 * @param src Source array of BF16 values
 * @param dst Destination array for FP32 values (must be pre-allocated)
 * @param n Number of elements to convert
 */
void bf16_to_f32_array(const bf16_t* src, float* dst, int n);

// ============================================================================
// Inline Function Implementations
// ============================================================================

/**
 * Scalar FP32 to BF16 conversion (inline implementation)
 *
 * Software implementation: truncate mantissa with round-to-nearest-even
 */
static inline bf16_t f32_to_bf16(float x) {
    // Interpret float as uint32
    uint32_t u;
    __builtin_memcpy(&u, &x, sizeof(float));

    // Round to nearest even (RNE)
    // Add rounding bias: 0x7FFF + LSB of the result
    uint32_t rounding_bias = 0x7FFF + ((u >> 16) & 1);
    u += rounding_bias;

    // Truncate to BF16 by taking upper 16 bits
    return (bf16_t)(u >> 16);
}

/**
 * Scalar BF16 to FP32 conversion (inline implementation)
 *
 * Lossless: just pad mantissa with zeros
 */
static inline float bf16_to_f32(bf16_t x) {
    // Shift left 16 bits to restore FP32 format
    uint32_t u = ((uint32_t)x) << 16;

    // Interpret as float
    float result;
    __builtin_memcpy(&result, &u, sizeof(float));
    return result;
}

#ifdef __cplusplus
}
#endif

#endif // BF16_UTIL_H

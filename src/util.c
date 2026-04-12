/*
 * Code for Fast k-Nearest Neighbour Search via Prioritized DCI
 * 
 * This code implements the method described in the Prioritized DCI paper, 
 * which can be found at https://arxiv.org/abs/1703.00440
 * 
 * This file is a part of the Dynamic Continuous Indexing reference 
 * implementation.
 * 
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. 
 * 
 * Copyright (C) 2017    Ke Li
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
//#include <malloc.h>
#include <math.h>
#include "util.h"
#include <unistd.h>
#include<immintrin.h>
#include <x86intrin.h>
#include <cblas.h>
#include "bf16_util.h"

// Assuming column-major layout, computes A^T * B. A is K x M, B is K x N, and C is M x N.
// BF16 version: inputs are BF16, output is BF16, but computation uses FP32 accumulation
void matmul(const int M, const int N, const int K, const bf16_t* const A, const bf16_t* const B, float* const C) {
#if defined(__AVX512BF16__)
    // Custom BF16 matrix multiply with dual accumulators for ILP
    // C[i,j] = A[:,i]^T * B[:,j] where A is K×M, B is K×N
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Dual accumulators for ILP
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            int k = 0;

            // Process 64 elements per iteration (2x 32-wide dpbf16)
            for (; k + 63 < K; k += 64) {
                __m512bh a0 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(&A[k + i*K]));
                __m512bh b0 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(&B[k + j*K]));
                __m512bh a1 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(&A[k + 32 + i*K]));
                __m512bh b1 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(&B[k + 32 + j*K]));
                acc0 = _mm512_dpbf16_ps(acc0, a0, b0);
                acc1 = _mm512_dpbf16_ps(acc1, a1, b1);
            }

            // Merge and handle one more 32-wide chunk
            __m512 acc = _mm512_add_ps(acc0, acc1);
            for (; k + 31 < K; k += 32) {
                __m512bh a_vec = (__m512bh)_mm512_loadu_epi16((const __m512i*)(&A[k + i*K]));
                __m512bh b_vec = (__m512bh)_mm512_loadu_epi16((const __m512i*)(&B[k + j*K]));
                acc = _mm512_dpbf16_ps(acc, a_vec, b_vec);
            }

            // Masked tail for remaining 0-31 elements - reuse accumulator!
            int rem = K - k;
            if (rem > 0) {
                // Use 64-bit shift to avoid UB if rem==32
                __mmask32 km = (rem >= 32) ? (__mmask32)0xFFFFFFFFu
                                           : (__mmask32)(((uint64_t)1u << rem) - 1u);
                __m512i av = _mm512_maskz_loadu_epi16(km, (const void*)(&A[k + i*K]));
                __m512i bv = _mm512_maskz_loadu_epi16(km, (const void*)(&B[k + j*K]));
                __m512bh ab = (__m512bh)av;
                __m512bh bb = (__m512bh)bv;
                acc = _mm512_dpbf16_ps(acc, ab, bb);
            }

            // Store result as BF16
            C[i + j*M] = _mm512_reduce_add_ps(acc);
        }
    }
#else
    // Software fallback: convert to FP32, use SGEMM, convert back
    float* A_f32 = (float*)malloc(K * M * sizeof(float));
    float* B_f32 = (float*)malloc(K * N * sizeof(float));
    float* C_f32 = (float*)malloc(M * N * sizeof(float));

    bf16_to_f32_array(A, A_f32, K * M);
    bf16_to_f32_array(B, B_f32, K * N);

    const char TRANSA = 'T';
    const char TRANSB = 'N';
    const float ALPHA = 1.;
    const float BETA = 0.;
    const int LDA = K;
    const int LDB = K;
    const int LDC = M;
    SGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A_f32, &LDA, B_f32, &LDB, &BETA, C_f32, &LDC);

    f32_to_bf16_array(C_f32, C, M * N);

    free(A_f32);
    free(B_f32);
    free(C_f32);
#endif
}

float vecmul(const bf16_t* const x, const bf16_t* const y, const int k) {
#if defined(__AVX512BF16__)
    // Dual accumulators for ILP (hides dpbf16 latency)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;

    // Process 64 BF16 elements per iteration (2x 32-wide dpbf16)
    for (; i + 63 < k; i += 64) {
        __m512bh x0 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(x + i));
        __m512bh y0 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(y + i));
        __m512bh x1 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(x + i + 32));
        __m512bh y1 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(y + i + 32));
        acc0 = _mm512_dpbf16_ps(acc0, x0, y0);
        acc1 = _mm512_dpbf16_ps(acc1, x1, y1);
    }

    // Merge and handle one more 32-wide chunk
    __m512 acc = _mm512_add_ps(acc0, acc1);
    for (; i + 31 < k; i += 32) {
        __m512bh X = (__m512bh)_mm512_loadu_epi16((const __m512i*)(x + i));
        __m512bh Y = (__m512bh)_mm512_loadu_epi16((const __m512i*)(y + i));
        acc = _mm512_dpbf16_ps(acc, X, Y);
    }

    // Masked tail for remaining 0-31 elements (dpbf16 with zero-fill)
    int rem = k - i;
    if (rem > 0) {
        // Use 64-bit shift to avoid UB if rem==32 ever happens
        __mmask32 km = (rem >= 32) ? (__mmask32)0xFFFFFFFFu
                                : (__mmask32)(((uint64_t)1u << rem) - 1u);

        __m512i xv = _mm512_maskz_loadu_epi16(km, (const void*)(x + i));
        __m512i yv = _mm512_maskz_loadu_epi16(km, (const void*)(y + i));
        __m512bh xb = (__m512bh)xv;
        __m512bh yb = (__m512bh)yv;
        acc = _mm512_dpbf16_ps(acc, xb, yb);
    }

    float result = _mm512_reduce_add_ps(acc);
    return result;
#else
    // Software fallback: convert BF16 to FP32 and compute
    float inner_prod = 0.0;
    for (int i = 0; i < k; i++) {
        inner_prod += bf16_to_f32(x[i]) * bf16_to_f32(y[i]);
    }
    return inner_prod;
#endif
}

float transform_compute_dist(const bf16_t* const vec1, const bf16_t* const vec2, const int dim, const float max_sq_norm, const float sq_norm1, const float sq_norm2) {
    float dots = vecmul(vec1, vec2, dim);
    float sq_dist = max_sq_norm - dots - sqrt((max_sq_norm - sq_norm1)*(max_sq_norm - sq_norm2));
    return sq_dist;
}

float transform_compute_dist_query(const bf16_t* const vec1, const bf16_t* const vec2, const int dim) {
    float sudo_dist = vecmul(vec1, vec2, dim);
    return -1*sudo_dist;
}

float compute_dist(const bf16_t* const vec1, const bf16_t* const vec2, const int dim) {
#if defined(__AVX512BF16__)
    // Dual accumulators for each of 3 terms (dot, norm1, norm2) - 6 total for ILP
    __m512 acc_dot0 = _mm512_setzero_ps();
    __m512 acc_dot1 = _mm512_setzero_ps();
    __m512 acc_norm1_0 = _mm512_setzero_ps();
    __m512 acc_norm1_1 = _mm512_setzero_ps();
    __m512 acc_norm2_0 = _mm512_setzero_ps();
    __m512 acc_norm2_1 = _mm512_setzero_ps();
    int i = 0;

    // Process 64 BF16 elements per iteration (2x 32-wide × 3 dpbf16)
    for (; i + 63 < dim; i += 64) {
        __m512bh v1_0 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(vec1 + i));
        __m512bh v2_0 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(vec2 + i));
        __m512bh v1_1 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(vec1 + i + 32));
        __m512bh v2_1 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(vec2 + i + 32));

        acc_dot0 = _mm512_dpbf16_ps(acc_dot0, v1_0, v2_0);
        acc_dot1 = _mm512_dpbf16_ps(acc_dot1, v1_1, v2_1);
        acc_norm1_0 = _mm512_dpbf16_ps(acc_norm1_0, v1_0, v1_0);
        acc_norm1_1 = _mm512_dpbf16_ps(acc_norm1_1, v1_1, v1_1);
        acc_norm2_0 = _mm512_dpbf16_ps(acc_norm2_0, v2_0, v2_0);
        acc_norm2_1 = _mm512_dpbf16_ps(acc_norm2_1, v2_1, v2_1);
    }

    // Merge and handle one more 32-wide chunk
    __m512 acc_dot = _mm512_add_ps(acc_dot0, acc_dot1);
    __m512 acc_norm1 = _mm512_add_ps(acc_norm1_0, acc_norm1_1);
    __m512 acc_norm2 = _mm512_add_ps(acc_norm2_0, acc_norm2_1);

    for (; i + 31 < dim; i += 32) {
        __m512bh v1 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(vec1 + i));
        __m512bh v2 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(vec2 + i));
        acc_dot = _mm512_dpbf16_ps(acc_dot, v1, v2);
        acc_norm1 = _mm512_dpbf16_ps(acc_norm1, v1, v1);
        acc_norm2 = _mm512_dpbf16_ps(acc_norm2, v2, v2);
    }

    // Masked tail for remaining 0-31 elements - reuse accumulators!
    int rem = dim - i;
    if (rem > 0) {
        // Use 64-bit shift to avoid UB if rem==32
        __mmask32 km = (rem >= 32) ? (__mmask32)0xFFFFFFFFu
                                   : (__mmask32)(((uint64_t)1u << rem) - 1u);
        __m512i v1_i = _mm512_maskz_loadu_epi16(km, (const void*)(vec1 + i));
        __m512i v2_i = _mm512_maskz_loadu_epi16(km, (const void*)(vec2 + i));
        __m512bh v1_b = (__m512bh)v1_i;
        __m512bh v2_b = (__m512bh)v2_i;

        acc_dot = _mm512_dpbf16_ps(acc_dot, v1_b, v2_b);
        acc_norm1 = _mm512_dpbf16_ps(acc_norm1, v1_b, v1_b);
        acc_norm2 = _mm512_dpbf16_ps(acc_norm2, v2_b, v2_b);
    }

    // Single reduce at the end
    float dot = _mm512_reduce_add_ps(acc_dot);
    float norm1 = _mm512_reduce_add_ps(acc_norm1);
    float norm2 = _mm512_reduce_add_ps(acc_norm2);

    // ||v1 - v2||² = ||v1||² + ||v2||² - 2·(v1·v2)
    float sq_dist = norm1 + norm2 - 2.0f * dot;
    return sqrtf(sq_dist);
#else
    // Software fallback
    float sq_dist = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = bf16_to_f32(vec1[i]) - bf16_to_f32(vec2[i]);
        sq_dist += diff * diff;
    }
    return sqrtf(sq_dist);
#endif
}

int find_min_distance_simd(const float* distances, int num_points, float* min_dist) {
    if (num_points == 0) {
        *min_dist = FLT_MAX;
        return -1;
    }

    if (num_points == 1) {
        *min_dist = distances[0];
        return 0;
    }

    int i = 0;
    __m256 vmin = _mm256_set1_ps(FLT_MAX);
    __m256i vindices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i vmin_indices = vindices;
    __m256i vstep = _mm256_set1_epi32(8);

    // Process 8 values at a time
    for (; i + 7 < num_points; i += 8) {
        __m256 v = _mm256_loadu_ps(distances + i);
        __m256 cmp = _mm256_cmp_ps(v, vmin, _CMP_LT_OQ);

        vmin = _mm256_min_ps(vmin, v);
        vmin_indices = _mm256_castps_si256(
            _mm256_blendv_ps(
                _mm256_castsi256_ps(vmin_indices),
                _mm256_castsi256_ps(vindices),
                cmp
            )
        );

        vindices = _mm256_add_epi32(vindices, vstep);
    }

    // Extract minimums
    float tmp_vals[8];
    int tmp_indices[8];
    _mm256_storeu_ps(tmp_vals, vmin);
    _mm256_storeu_si256((__m256i*)tmp_indices, vmin_indices);

    *min_dist = tmp_vals[0];
    int min_idx = tmp_indices[0];

    for (int j = 1; j < 8; j++) {
        if (tmp_vals[j] < *min_dist) {
            *min_dist = tmp_vals[j];
            min_idx = tmp_indices[j];
        }
    }

    // Handle remainder
    for (; i < num_points; i++) {
        if (distances[i] < *min_dist) {
            *min_dist = distances[i];
            min_idx = i;
        }
    }

    return min_idx;
}

void compute_distances_matrix_blas(
    const bf16_t* query,
    const bf16_t* points_matrix,
    int num_points,
    int dim,
    const float* points_sq_norms,
    float* distances)
{
#if defined(__AVX512BF16__)
    // Compute ||query||^2
    float query_sq_norm = 0.0f;
    {
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        int i = 0;

        for (; i + 63 < dim; i += 64) {
            __m512bh q0 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(query + i));
            __m512bh q1 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(query + i + 32));
            acc0 = _mm512_dpbf16_ps(acc0, q0, q0);
            acc1 = _mm512_dpbf16_ps(acc1, q1, q1);
        }

        __m512 acc = _mm512_add_ps(acc0, acc1);

        for (; i + 31 < dim; i += 32) {
            __m512bh q = (__m512bh)_mm512_loadu_epi16((const __m512i*)(query + i));
            acc = _mm512_dpbf16_ps(acc, q, q);
        }

        int rem = dim - i;
        if (rem > 0) {
            __mmask32 km = (rem >= 32) ? (__mmask32)0xFFFFFFFFu
                                       : (__mmask32)(((uint64_t)1u << rem) - 1u);
            __m512i qv = _mm512_maskz_loadu_epi16(km, (const void*)(query + i));
            acc = _mm512_dpbf16_ps(acc, (__m512bh)qv, (__m512bh)qv);
        }

        query_sq_norm = _mm512_reduce_add_ps(acc);
    }

    // Distances
    for (int p = 0; p < num_points; p++) {
        const bf16_t* point = points_matrix + (size_t)p * (size_t)dim;

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        int i = 0;

        for (; i + 63 < dim; i += 64) {
            __m512bh q0 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(query + i));
            __m512bh x0 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(point + i));
            __m512bh q1 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(query + i + 32));
            __m512bh x1 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(point + i + 32));
            acc0 = _mm512_dpbf16_ps(acc0, q0, x0);
            acc1 = _mm512_dpbf16_ps(acc1, q1, x1);
        }

        __m512 acc = _mm512_add_ps(acc0, acc1);

        for (; i + 31 < dim; i += 32) {
            __m512bh q = (__m512bh)_mm512_loadu_epi16((const __m512i*)(query + i));
            __m512bh x = (__m512bh)_mm512_loadu_epi16((const __m512i*)(point + i));
            acc = _mm512_dpbf16_ps(acc, q, x);
        }

        int rem = dim - i;
        if (rem > 0) {
            __mmask32 km = (rem >= 32) ? (__mmask32)0xFFFFFFFFu
                                       : (__mmask32)(((uint64_t)1u << rem) - 1u);
            __m512i qv = _mm512_maskz_loadu_epi16(km, (const void*)(query + i));
            __m512i xv = _mm512_maskz_loadu_epi16(km, (const void*)(point + i));
            acc = _mm512_dpbf16_ps(acc, (__m512bh)qv, (__m512bh)xv);
        }

        float dot = _mm512_reduce_add_ps(acc);

        float sq_dist = query_sq_norm + points_sq_norms[p] - 2.0f * dot;
        sq_dist = fmaxf(sq_dist, 0.0f);   // avoid sqrt of tiny negative
        distances[p] = sqrtf(sq_dist);
    }
#else
    // Software fallback: convert to FP32
    float* query_f32 = (float*)malloc(dim * sizeof(float));
    float* points_f32 = (float*)malloc(num_points * dim * sizeof(float));

    bf16_to_f32_array(query, query_f32, dim);
    bf16_to_f32_array(points_matrix, points_f32, num_points * dim);

    float query_sq_norm = cblas_sdot(dim, query_f32, 1, query_f32, 1);
    float* dots = (float*)malloc(num_points * sizeof(float));

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                num_points, dim,
                1.0f, points_f32, dim,
                query_f32, 1,
                0.0f, dots, 1);

    __m256 vquery_norm = _mm256_set1_ps(query_sq_norm);
    __m256 vtwo = _mm256_set1_ps(2.0f);

    for (; i + 7 < num_points; i += 8) {
        __m256 vpoint_norms = _mm256_loadu_ps(points_sq_norms + i);
        __m256 vdots = _mm256_loadu_ps(dots + i);

        __m256 vsq_dist = _mm256_add_ps(vquery_norm, vpoint_norms);
        vsq_dist = _mm256_fnmadd_ps(vtwo, vdots, vsq_dist);

        __m256 vdist = _mm256_sqrt_ps(vsq_dist);
        _mm256_storeu_ps(distances + i, vdist);
    }

    for (; i < num_points; i++) {
        float sq_dist = query_sq_norm + points_sq_norms[i] - 2.0f * dots[i];
        distances[i] = sqrtf(sq_dist);
    }

    free(dots);
    free(query_f32);
    free(points_f32);
#endif
}

void compute_distances_transform_blas(
    const bf16_t* query,
    const bf16_t* points_matrix,
    int num_points,
    int dim,
    float max_sq_norm,
    float query_sq_norm,
    const float* points_sq_norms,
    float* distances)
{
#if defined(__AVX512BF16__)
    // Precompute (max - ||q||^2) once and clamp for safety
    float a = fmaxf(max_sq_norm - query_sq_norm, 0.0f);

    for (int p = 0; p < num_points; p++) {
        const bf16_t* point = points_matrix + (size_t)p * (size_t)dim;

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        int i = 0;

        // 64 elements per iter
        for (; i + 63 < dim; i += 64) {
            __m512bh q0  = (__m512bh)_mm512_loadu_epi16((const __m512i*)(query + i));
            __m512bh x0  = (__m512bh)_mm512_loadu_epi16((const __m512i*)(point + i));
            __m512bh q1  = (__m512bh)_mm512_loadu_epi16((const __m512i*)(query + i + 32));
            __m512bh x1  = (__m512bh)_mm512_loadu_epi16((const __m512i*)(point + i + 32));
            acc0 = _mm512_dpbf16_ps(acc0, q0, x0);
            acc1 = _mm512_dpbf16_ps(acc1, q1, x1);
        }

        __m512 acc = _mm512_add_ps(acc0, acc1);

        // one more 32-wide
        for (; i + 31 < dim; i += 32) {
            __m512bh qv = (__m512bh)_mm512_loadu_epi16((const __m512i*)(query + i));
            __m512bh xv = (__m512bh)_mm512_loadu_epi16((const __m512i*)(point + i));
            acc = _mm512_dpbf16_ps(acc, qv, xv);
        }

        // masked tail (0..31)
        int rem = dim - i;
        if (rem > 0) {
            __mmask32 km = (rem >= 32) ? (__mmask32)0xFFFFFFFFu
                                       : (__mmask32)(((uint64_t)1u << rem) - 1u);

            __m512i q16 = _mm512_maskz_loadu_epi16(km, (const void*)(query + i));
            __m512i x16 = _mm512_maskz_loadu_epi16(km, (const void*)(point + i));
            acc = _mm512_dpbf16_ps(acc, (__m512bh)q16, (__m512bh)x16);
        }

        float dot = _mm512_reduce_add_ps(acc);

        float b = fmaxf(max_sq_norm - points_sq_norms[p], 0.0f);
        float sqrt_term = sqrtf(a * b);

        distances[p] = max_sq_norm - dot - sqrt_term;
    }
#else
    // Software fallback: convert to FP32
    float* query_f32 = (float*)malloc(dim * sizeof(float));
    float* points_f32 = (float*)malloc(num_points * dim * sizeof(float));

    bf16_to_f32_array(query, query_f32, dim);
    bf16_to_f32_array(points_matrix, points_f32, num_points * dim);

    float* dots = (float*)malloc(num_points * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                num_points, dim,
                1.0f, points_f32, dim,
                query_f32, 1,
                0.0f, dots, 1);

    // Transform formula
    __m256 vmax = _mm256_set1_ps(max_sq_norm);
    __m256 vquery_term = _mm256_set1_ps(max_sq_norm - query_sq_norm);

    int i = 0;
    for (; i + 7 < num_points; i += 8) {
        __m256 vdots = _mm256_loadu_ps(dots + i);
        __m256 vpoint_norms = _mm256_loadu_ps(points_sq_norms + i);

        __m256 vpoint_term = _mm256_sub_ps(vmax, vpoint_norms);
        __m256 vprod = _mm256_mul_ps(vquery_term, vpoint_term);
        __m256 vsqrt_term = _mm256_sqrt_ps(vprod);

        __m256 vdist = _mm256_sub_ps(vmax, vdots);
        vdist = _mm256_sub_ps(vdist, vsqrt_term);

        _mm256_storeu_ps(distances + i, vdist);
    }

    for (; i < num_points; i++) {
        float sqrt_term = sqrtf((max_sq_norm - query_sq_norm) *
                               (max_sq_norm - points_sq_norms[i]));
        distances[i] = max_sq_norm - dots[i] - sqrt_term;
    }

    free(dots);
#endif
}

void compute_distances_matrix_portable(
    const bf16_t* query,
    const bf16_t* points_matrix,
    int num_points,
    int dim,
    float* distances)
{
#if defined(__AVX512BF16__)
    // AVX512_BF16 implementation
    // Compute ||query||²
    __m512 query_acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 31 < dim; i += 32) {
        __m512bh q = (__m512bh)_mm512_loadu_epi16((const __m512i*)(query + i));
        query_acc = _mm512_dpbf16_ps(query_acc, q, q);
    }
    float query_sq_norm = _mm512_reduce_add_ps(query_acc);
    for (; i < dim; i++) {
        float val = bf16_to_f32(query[i]);
        query_sq_norm += val * val;
    }

    // For each point
    for (int p = 0; p < num_points; p++) {
        const bf16_t* point = points_matrix + p * dim;

        __m512 dot_acc = _mm512_setzero_ps();
        __m512 norm_acc = _mm512_setzero_ps();

        int j = 0;
        for (; j + 31 < dim; j += 32) {
            __m512bh q = (__m512bh)_mm512_loadu_epi16((const __m512i*)(query + j));
            __m512bh pt = (__m512bh)_mm512_loadu_epi16((const __m512i*)(point + j));

            dot_acc = _mm512_dpbf16_ps(dot_acc, q, pt);
            norm_acc = _mm512_dpbf16_ps(norm_acc, pt, pt);
        }

        float dot_product = _mm512_reduce_add_ps(dot_acc);
        float point_sq_norm = _mm512_reduce_add_ps(norm_acc);

        for (; j < dim; j++) {
            float q_val = bf16_to_f32(query[j]);
            float p_val = bf16_to_f32(point[j]);
            dot_product += q_val * p_val;
            point_sq_norm += p_val * p_val;
        }

        float sq_dist = query_sq_norm + point_sq_norm - 2.0f * dot_product;
        distances[p] = sqrtf(sq_dist);
    }
#else
    // Software fallback - convert to FP32 and compute
    float* query_fp32 = (float*)malloc(dim * sizeof(float));
    float* points_fp32 = (float*)malloc(num_points * dim * sizeof(float));

    bf16_to_f32_array(query, query_fp32, dim);
    bf16_to_f32_array(points_matrix, points_fp32, num_points * dim);

    // Compute ||query||²
    float query_sq_norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        query_sq_norm += query_fp32[i] * query_fp32[i];
    }

    // For each point
    for (int p = 0; p < num_points; p++) {
        const float* point = points_fp32 + p * dim;

        float dot_product = 0.0f;
        float point_sq_norm = 0.0f;

        for (int i = 0; i < dim; i++) {
            dot_product += query_fp32[i] * point[i];
            point_sq_norm += point[i] * point[i];
        }

        float sq_dist = query_sq_norm + point_sq_norm - 2.0f * dot_product;
        distances[p] = sqrtf(sq_dist);
    }

    free(query_fp32);
    free(points_fp32);
#endif
}

// ***********************************************************************
// !! This funtion is not implemented with BF16 intrinsics for simplicity
// ***********************************************************************
void gen_data(float* const data, const int ambient_dim, const int intrinsic_dim, const int num_points) {
    int i;
    bf16_t* latent_data;
    bf16_t* transformation;
    bf16_t* result_bf16;

    int ret = posix_memalign((void **)&latent_data, 64, sizeof(bf16_t)*intrinsic_dim*num_points);
    assert(ret == 0);
    ret = posix_memalign((void **)&transformation, 64, sizeof(bf16_t)*intrinsic_dim*ambient_dim);
    assert(ret == 0);
    ret = posix_memalign((void **)&result_bf16, 64, sizeof(bf16_t)*ambient_dim*num_points);
    assert(ret == 0);

    // Generate random data in FP32 first
    float* temp_latent = (float*)malloc(sizeof(float)*intrinsic_dim*num_points);
    float* temp_transformation = (float*)malloc(sizeof(float)*intrinsic_dim*ambient_dim);

    for (i = 0; i < intrinsic_dim*num_points; i++) {
        temp_latent[i] = 2 * drand48() - 1;
    }
    for (i = 0; i < intrinsic_dim*ambient_dim; i++) {
        temp_transformation[i] = 2 * drand48() - 1;
    }

    // Convert to BF16
    f32_to_bf16_array(temp_latent, latent_data, intrinsic_dim*num_points);
    f32_to_bf16_array(temp_transformation, transformation, intrinsic_dim*ambient_dim);

    free(temp_latent);
    free(temp_transformation);

    // Assuming column-major layout, transformation is intrisic_dim x ambient_dim,
    // latent_data is intrinsic_dim x num_points, result_bf16 is ambient_dim x num_points
    matmul(ambient_dim, num_points, intrinsic_dim, transformation, latent_data, result_bf16);

    // Convert result back to FP32 for output
    bf16_to_f32_array(result_bf16, data, ambient_dim*num_points);

    free(latent_data);
    free(transformation);
    free(result_bf16);
}

float rand_normal() {
    static float V1, V2, S;
    static int phase = 0;
    float X;

    if(phase == 0) {
        do {
            float U1 = drand48();
            float U2 = drand48();
            
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
            } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    return X;
}

// Print matrix assuming column-major layout
void print_matrix(const float* const data, const int num_rows, const int num_cols) {
    int i, j;
    for (i = 0; i < num_rows; i++) {
        for (j = 0; j < num_cols; j++) {
            printf("%.4f\t", data[i+j*num_rows]);
        }
        printf("\n");
    }
}

// Apply query transformation
void query_transform(
    const bf16_t* data,          // [num_points, dim] BF16
    int num_points,
    int dim,
    float* projs,                // [num_points, num_idx] FP32 (in/out)
    int num_idx)
{
    for (int i = 0; i < num_points; i++) {
        const bf16_t* x = data + (size_t)i * (size_t)dim;

        // Compute ||x||^2 in FP32
        float norm2 = 0.0f;
        
#if defined(__AVX512BF16__)
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        int k = 0;

        // 64 BF16 at a time using 2 accumulators
        for (; k + 63 < dim; k += 64) {
            __m512bh v0 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(x + k));
            __m512bh v1 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(x + k + 32));
            acc0 = _mm512_dpbf16_ps(acc0, v0, v0);
            acc1 = _mm512_dpbf16_ps(acc1, v1, v1);
        }

        __m512 acc = _mm512_add_ps(acc0, acc1);

        // one more full 32-wide chunk
        for (; k + 31 < dim; k += 32) {
            __m512bh v = (__m512bh)_mm512_loadu_epi16((const __m512i*)(x + k));
            acc = _mm512_dpbf16_ps(acc, v, v);
        }

        // masked tail 0..31
        int rem = dim - k;
        if (rem > 0) {
            __mmask32 km = (rem >= 32) ? (__mmask32)0xFFFFFFFFu
                                       : (__mmask32)(((uint64_t)1u << rem) - 1u);
            __m512i vv = _mm512_maskz_loadu_epi16(km, (const void*)(x + k));
            acc = _mm512_dpbf16_ps(acc, (__m512bh)vv, (__m512bh)vv);
        }

        norm2 = _mm512_reduce_add_ps(acc);
#else
        for (int k = 0; k < dim; k++) {
            float v = bf16_to_f32(x[k]);
            norm2 += v * v;
        }
#endif

        // Safety: avoid divide-by-zero / NaNs
        assert(norm2 > 0.0f);
        float inv_norm = 1.0f / sqrtf(norm2);

        // Scale projs for this point: projs[i, :] *= inv_norm
        float* row = projs + (size_t)i * (size_t)num_idx;
        for (int j = 0; j < num_idx; j++) {
            row[j] *= inv_norm;
        }
    }
}

int compare_float(const void *a, const void *b, void *array) {
    int index1 = *(const int *)a;
    int index2 = *(const int *)b;
    const float *arr = (const float *)array;

    return (arr[index1] > arr[index2]) - (arr[index1] < arr[index2]);
}

int compare_float_r(const void *a, const void *b, void *array) {
    int index1 = *(const int *)a;
    int index2 = *(const int *)b;
    const float *arr = (const float *)array;

    return (arr[index1] < arr[index2]) - (arr[index1] > arr[index2]);
}

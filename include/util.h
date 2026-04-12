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

#ifndef UTIL_H
#define UTIL_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include "bf16_util.h"

#ifdef USE_MKL
#define SGEMM sgemm
#else
#define SGEMM sgemm_
#endif  // USE_MKL

// BLAS native Fortran interface
extern void SGEMM(const char* const transa, const char* const transb, const int* const m, const int* const n, const int* const k, const float* const alpha, const float* const A, const int* const lda, const float* const B, const int* const ldb, const float* const beta, float* const C, const int* const ldc);

// BF16 computation functions
float vecmul(const bf16_t* const x, const bf16_t* const y, const int k);

void matmul(const int M, const int N, const int K, const bf16_t* const A, const bf16_t* const B, float* const C);

void gen_data(float* const data, const int ambient_dim, const int intrinsic_dim, const int num_points);

// SIMD min-finding
int find_min_distance_simd(const float* distances, int num_points, float* min_dist);

// BF16-accelerated distance computations
void compute_distances_matrix_blas(const bf16_t* query, const bf16_t* points_matrix,
    int num_points, int dim, const float* points_sq_norms, float* distances);

void compute_distances_transform_blas(const bf16_t* query, const bf16_t* points_matrix,
    int num_points, int dim, float max_sq_norm, float query_sq_norm,
    const float* points_sq_norms, float* distances);

// Portable fallback
void compute_distances_matrix_portable(const bf16_t* query, const bf16_t* points_matrix,
    int num_points, int dim, float* distances);

float transform_compute_dist(const bf16_t* const vec1, const bf16_t* const vec2, const int dim, const float max_sq_norm, const float sq_norm1, const float sq_norm2);

float transform_compute_dist_query(const bf16_t* const vec1, const bf16_t* const vec2, const int dim);

float compute_dist(const bf16_t* const vec1, const bf16_t* const vec2, const int dim);

float rand_normal();

void print_matrix(const float* const data, const int num_rows, const int num_cols);

void query_transform(const bf16_t* data, const int num_points, const int dim, float* const projs, const int num_idx);

int compare_float(const void *a, const void *b, void *array);
int compare_float_r(const void *a, const void *b, void *array);

#ifdef __cplusplus
}
#endif

#endif // UTIL_H

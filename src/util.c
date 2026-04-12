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

// Assuming column-major layout, computes A^T * B. A is K x M, B is K x N, and C is M x N. 
void matmul(const int M, const int N, const int K, const float* const A, const float* const B, float* const C) {
    const char TRANSA = 'T';
    const char TRANSB = 'N';
    const float ALPHA = 1.; 
    const float BETA = 0.; 
    const int LDA = K;
    const int LDB = K;
    const int LDC = M;
    SGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

float vecmul(const float* const x, const float* const y, const int k) {
    float inner_prod = 0.0;
    __m256 X, Y; // 256-bit values
	__m256 acc = _mm256_setzero_ps(); // set to (0, 0, 0, 0)
	float temp[8];
	long i;
	for (i = 0; i < k - 8; i += 8)
	{
		X = _mm256_loadu_ps(x + i); // load chunk of 8 floats
		Y = _mm256_loadu_ps(y + i);
		acc = _mm256_add_ps(acc, _mm256_mul_ps(X, Y));
	}
	_mm256_storeu_ps(&temp[0], acc); // store acc into an array of floats
	inner_prod = temp[0] + temp[1] + temp[2] + temp[3]  + temp[4]  + temp[5]  + temp[6]  + temp[7];
	// add the remaining values
	for (; i < k; i++)
		inner_prod += x[i] * y[i];
	return inner_prod;
}

float transform_compute_dist(const float* const vec1, const float* const vec2, const int dim, const float max_sq_norm, const float sq_norm1, const float sq_norm2) {
    int i;
    float sq_dist = 0.0;
    float dots = 0.0;
    // for (i = 0; i < dim; i++) {
    //     dots += (vec1[i])*(vec2[i]);
    // }
    dots = vecmul(vec1, vec2, dim);
    sq_dist = max_sq_norm - dots - sqrt((max_sq_norm - sq_norm1)*(max_sq_norm - sq_norm2));
    return sq_dist;
}

float transform_compute_dist_query(const float* const vec1, const float* const vec2, const int dim) {
    float sudo_dist = 0.0;
    sudo_dist = vecmul(vec1, vec2, dim);

    return -1*sudo_dist;
}

float compute_dist(const float* const vec1, const float* const vec2, const int dim) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;

    for (; i + 7 < dim; i += 8) {
        __m256 x = _mm256_loadu_ps(vec1 + i);
        __m256 y = _mm256_loadu_ps(vec2 + i);
        __m256 diff = _mm256_sub_ps(x, y);
        __m256 sq = _mm256_mul_ps(diff, diff);
        acc = _mm256_add_ps(acc, sq);
    }

    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < dim; ++i) {
        float diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }

    return sqrt(sum);
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
    const float* query,
    const float* points_matrix,
    int num_points,
    int dim,
    const float* points_sq_norms,
    float* distances)
{
    // Compute ||query||²
    float query_sq_norm = cblas_sdot(dim, query, 1, query, 1);

    // Compute all dot products: dots = points_matrix @ query
    float* dots = (float*)malloc(num_points * sizeof(float));

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                num_points, dim,
                1.0f, points_matrix, dim,
                query, 1,
                0.0f, dots, 1);

    // Compute distances: ||q - p||² = ||q||² + ||p||² - 2·q·p
    __m256 vquery_norm = _mm256_set1_ps(query_sq_norm);
    __m256 vtwo = _mm256_set1_ps(2.0f);

    int i = 0;
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
}

void compute_distances_transform_blas(
    const float* query,
    const float* points_matrix,
    int num_points,
    int dim,
    float max_sq_norm,
    float query_sq_norm,
    const float* points_sq_norms,
    float* distances)
{
    // Compute all dot products
    float* dots = (float*)malloc(num_points * sizeof(float));

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                num_points, dim,
                1.0f, points_matrix, dim,
                query, 1,
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
}

void compute_distances_matrix_portable(
    const float* query,
    const float* points_matrix,
    int num_points,
    int dim,
    float* distances)
{
    // Compute ||query||²
    float query_sq_norm = 0.0f;
    {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < dim; i += 8) {
            __m256 q = _mm256_loadu_ps(query + i);
            acc = _mm256_fmadd_ps(q, q, acc);
        }

        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        query_sq_norm = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                        tmp[4] + tmp[5] + tmp[6] + tmp[7];

        for (; i < dim; i++) {
            query_sq_norm += query[i] * query[i];
        }
    }

    // For each point
    for (int p = 0; p < num_points; p++) {
        const float* point = points_matrix + p * dim;

        float dot_product = 0.0f;
        float point_sq_norm = 0.0f;

        __m256 dot_acc = _mm256_setzero_ps();
        __m256 norm_acc = _mm256_setzero_ps();

        int i = 0;
        for (; i + 7 < dim; i += 8) {
            __m256 q = _mm256_loadu_ps(query + i);
            __m256 p_vec = _mm256_loadu_ps(point + i);

            dot_acc = _mm256_fmadd_ps(q, p_vec, dot_acc);
            norm_acc = _mm256_fmadd_ps(p_vec, p_vec, norm_acc);
        }

        float tmp[8];
        _mm256_storeu_ps(tmp, dot_acc);
        dot_product = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                      tmp[4] + tmp[5] + tmp[6] + tmp[7];

        _mm256_storeu_ps(tmp, norm_acc);
        point_sq_norm = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                        tmp[4] + tmp[5] + tmp[6] + tmp[7];

        for (; i < dim; i++) {
            dot_product += query[i] * point[i];
            point_sq_norm += point[i] * point[i];
        }

        float sq_dist = query_sq_norm + point_sq_norm - 2.0f * dot_product;
        distances[p] = sqrtf(sq_dist);
    }
}

void gen_data(float* const data, const int ambient_dim, const int intrinsic_dim, const int num_points) {
    int i;
    float* latent_data;
    float* transformation;

    int ret = posix_memalign((void **)&latent_data, 32, sizeof(float)*intrinsic_dim*num_points);
    assert(ret == 0);
    ret = posix_memalign((void **)&transformation, 32, sizeof(float)*intrinsic_dim*ambient_dim);
    assert(ret == 0);
    
    for (i = 0; i < intrinsic_dim*num_points; i++) {
        latent_data[i] = 2 * drand48() - 1;
    }
    for (i = 0; i < intrinsic_dim*ambient_dim; i++) {
        transformation[i] = 2 * drand48() - 1;
    }
    // Assuming column-major layout, transformation is intrisic_dim x ambient_dim, 
    // latent_data is intrinsic_dim x num_points, data is ambient_dim x num_points
    matmul(ambient_dim, num_points, intrinsic_dim, transformation, latent_data, data);
    free(latent_data);
    free(transformation);
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
void query_transform(const float* const data, const int num_points, const int dim, float* const projs, const int num_idx) {
    int i, j;
    float norm_list[num_points];
    float norm = 0.0;
    for (i = 0; i < num_points; i++) {
        for (j = 0; j < dim; j++) {
            norm += data[j+i*dim] * data[j+i*dim];
        }
        norm_list[i] = norm;
        norm = 0.0;
    }
    assert(norm_list[0] > 0.0);
    for (i = 0; i < num_points; i++) {
        for (j = 0; j < num_idx; j++) {
            projs[j+i*num_idx] = projs[j+i*num_idx]/sqrt(norm_list[i]);
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

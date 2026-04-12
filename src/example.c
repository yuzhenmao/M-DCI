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

//#include <malloc.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dci.h"
#include "util.h"
#include "hashtable_p.h"

int main(int argc, char **argv) {
    // srand48(time(NULL));
    srand48(0);  // For reproducibility (but not for performance testing)

    int j, k;

    int dim = 5000;
    int intrinsic_dim = 50;
    int num_points = 50000;
    int num_queries = 5;
    int num_neighbours = 8;  // The k in k-NN

    // Guide for tuning hyperparameters:

    // num_comp_indices trades off accuracy vs. construction and query time -
    // high values lead to more accurate results, but slower construction and
    // querying num_simp_indices trades off accuracy vs. construction and query
    // time - high values lead to more accurate results, but slower construction
    // and querying; if num_simp_indices is increased, may need to increase
    // num_comp_indices num_levels trades off construction time vs. query time -
    // higher values lead to faster querying, but slower construction; if
    // num_levels is increased, may need to increase query_field_of_view and
    // construction_field_of_view construction_field_of_view trades off
    // accuracy/query time vs. construction time - higher values lead to
    // *slightly* more accurate results and/or *slightly* faster querying, but
    // *slightly* slower construction construction_prop_to_retrieve trades off
    // acrruacy vs. construction time - higher values lead to *slightly* more
    // accurate results, but slower construction query_field_of_view trades off
    // accuracy vs. query time - higher values lead to more accurate results,
    // but *slightly* slower querying query_prop_to_retrieve trades off accuracy
    // vs. query time - higher values lead to more accurate results, but slower
    // querying

    int num_comp_indices = 2;
    int num_simp_indices = 7;
    int num_levels = 2;
    int construction_field_of_view = 10;
    float construction_prop_to_retrieve = 0.002;
    int query_field_of_view = 170;
    float query_prop_to_retrieve = 0.8;
    float promotion_prob = 0.3;   // less than one

    // Generate data
    // Assuming column-major layout, data is dim x num_points
    float *data;
    if (posix_memalign((void **) &data, 64, sizeof(float) * dim * (num_points + num_queries)) != 0) {
        perror("posix_memalign failed!");
    }
    // float* data = (float *)memalign(64,
    // sizeof(float)*dim*(num_points+num_queries));
    gen_data(data, dim, intrinsic_dim, num_points + num_queries);
    // Assuming column-major layout, query is dim x num_queries
    float *query = data + dim * ((long long int) num_points);

    // print_matrix(data, dim, num_points);

    bool transform = true;
    int parallel_level = 3;
    bool debug = true;

    dci dci_inst;
    dci_init(&dci_inst, dim, num_comp_indices, num_simp_indices, promotion_prob, num_points, transform, parallel_level, debug);

    // print_matrix(dci_inst.proj_vec, dim, num_comp_indices*num_simp_indices);

    dci_query_config construction_query_config;

    construction_query_config.blind = false;
    construction_query_config.num_to_visit = 5000;
    construction_query_config.num_to_retrieve = -1;
    construction_query_config.prop_to_visit = 1.0;
    construction_query_config.prop_to_retrieve = construction_prop_to_retrieve;
    construction_query_config.field_of_view = construction_field_of_view;
    construction_query_config.target_level = 0;
    long long* d_ids = NULL;

    // dci_add(&dci_inst, dim, num_points, data, num_levels,
    // construction_query_config);
    int to_exclude = 20;
    dci_add(&dci_inst, dim, to_exclude, data, 3,
            construction_query_config, d_ids, 0, NULL, NULL);
    printf("-- Now there are %d points in DCI-tree.\n", dci_inst.num_points);
    print_dci(&dci_inst);

    long long data_ids[to_exclude];


    // Query
    dci_query_config query_config;

    query_config.blind = false;
    query_config.num_to_visit = 5000;
    query_config.num_to_retrieve = -1;
    query_config.prop_to_visit = 1.0;
    query_config.prop_to_retrieve = query_prop_to_retrieve;
    query_config.field_of_view = query_field_of_view;
    query_config.target_level = 0;

    // Assuming column-major layout, matrix is of size num_neighbours x
    // num_queries
    int **nearest_neighbours = (int **) malloc(sizeof(int *) * num_queries);
    float **nearest_neighbour_dists =
            (float **) malloc(sizeof(float *) * num_queries);
    int *num_returned = (int *) malloc(sizeof(int) * num_queries);
    long long first_id;

    for (int i = 0; i < 10; i++) {
        first_id = dci_add(&dci_inst, dim, to_exclude, data + ((i + 1) * to_exclude) * dim, 4,
                                     construction_query_config, d_ids, 0, NULL, NULL);
        printf("-- Now there are %d points in DCI-tree.\n", dci_inst.num_points);
        print_dci(&dci_inst);

        for (j = 0; j < to_exclude; j++)
            data_ids[j] = first_id + j - (to_exclude / 4);

        long long dup_ids[to_exclude];
        int num_deleted = dci_delete(&dci_inst, to_exclude, data_ids, construction_query_config, dup_ids);
        printf("-- Now there are %d points in DCI-tree.\n", dci_inst.num_points);

        dci_query(&dci_inst, dim, num_queries, query, num_neighbours,
                  query_config, NULL, nearest_neighbours, nearest_neighbour_dists,
                  num_returned);
        for (j = 0; j < num_queries; j++) {
            printf("%d: ", j + 1);
            for (k = 0; k < num_returned[j]; k++) {
                printf("%d: %.4f, ", nearest_neighbours[j][k],
                       nearest_neighbour_dists[j][k]);
            }
            printf("%d: %.4f\n", nearest_neighbours[j][num_neighbours - 1],
                   nearest_neighbour_dists[j][num_neighbours - 1]);
        }
        for (j = 0; j < num_queries; j++) {
            free(nearest_neighbours[j]);
        }
        for (j = 0; j < num_queries; j++) {
            free(nearest_neighbour_dists[j]);
        }
    }

    // Delete most of the data
    for (j = 0; j < to_exclude - (to_exclude / 4) - 10; j++)
        data_ids[j] = j;
    
    for (; j < to_exclude - 10; j++)
        data_ids[j] = first_id + j + 10;
    
    long long dup_ids[to_exclude - 10];
    int num_deleted = dci_delete(&dci_inst, to_exclude - 10, data_ids, construction_query_config, dup_ids);
    printf("-- Now there are %d points in DCI-tree.\n", dci_inst.num_points);

    dci_query(&dci_inst, dim, num_queries, query, num_neighbours,
                query_config, NULL, nearest_neighbours, nearest_neighbour_dists,
                num_returned);
    for (j = 0; j < num_queries; j++) {
        printf("%d: ", j + 1);
        for (k = 0; k < num_returned[j]; k++) {
            printf("%d: %.4f, ", nearest_neighbours[j][k],
                    nearest_neighbour_dists[j][k]);
        }
        printf("%d: %.4f\n", nearest_neighbours[j][num_neighbours - 1],
                nearest_neighbour_dists[j][num_neighbours - 1]);
    }
    for (j = 0; j < num_queries; j++) {
        free(nearest_neighbours[j]);
    }
    for (j = 0; j < num_queries; j++) {
        free(nearest_neighbour_dists[j]);
    }

    // Print the tree
    print_dci(&dci_inst);
    

    // Free memory
    dci_free(&dci_inst);
    free(nearest_neighbours);
    free(nearest_neighbour_dists);
    free(num_returned);
    free(data);

    return 0;
}

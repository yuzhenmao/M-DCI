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
#include "dci.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include "btree_i.h"
#include "hashtable_d.h"
#include "hashtable_i.h"
#include "hashtable_p.h"
#include<immintrin.h>
#include <x86intrin.h>
#include <limits.h>
#include "util.h"

// Print the structure of the whole dci
void print_dci(dci* dci_inst) {
    int j;
    int num_levels = dci_inst->num_levels;
    int* num_points_on_level = dci_inst->num_points_on_level;
    additional_info* root = dci_inst->root;
    int idx = num_levels - 1;
    int num_points_on_upper_levels = 1, num_points_on_cur_levels = 0;
    btree_p* cur_tree;
    btree_p_search_res s;
    additional_info*** level_points = (additional_info***)malloc(sizeof(additional_info**)*(num_levels));

    while (idx >= 0) {
        num_points_on_cur_levels = num_points_on_level[idx];
        level_points[idx] = (additional_info**)malloc(
            sizeof(additional_info*) * (num_points_on_cur_levels));
        int hh = 0;
        for (j = 0; j < num_points_on_upper_levels; j++) {
            cur_tree = root->cell_indices;
            if (idx < num_levels - 1)
                cur_tree = level_points[idx + 1][j]->cell_indices;
            for (s = btree_p_first(cur_tree); !btree_p_is_end(cur_tree, s); s = btree_p_find_next(s)) {
                level_points[idx][hh++] = btree_p_valueof(s).info;
            }
        }
        num_points_on_upper_levels = num_points_on_cur_levels;
        idx -= 1;
    }
    for (idx = num_levels - 1; idx >= 0; idx--) {
        num_points_on_cur_levels = num_points_on_level[idx];
        if (idx > 0) {
            for (j = 0; j < num_points_on_cur_levels; j++) {
                printf("%d(%d)  ", level_points[idx][j]->id, level_points[idx][j]->num_finest_level_points[0]);
            }
        }
        else {
            for (j = 0; j < num_points_on_cur_levels; j++) {
                printf("%d(1)  ", level_points[idx][j]->id);
            }
        }
        printf("\n");
    }
    printf("#############\n");

    idx = num_levels - 1;
    while (idx >= 0) {
        free(level_points[idx]);
        idx -= 1;
    }
    free(level_points);
}

// Get anchor parent statistics for the finest level
void get_parent_stat(dci* dci_inst, int X, int interval, bool* parent_in_anchor_set, float* distance_ratios, bool* parent_consistency, float* min_dist_to_anchor_parents, float* max_dist_closest_parent_to_children) {
    int num_levels = dci_inst->num_levels;
    int* num_points_on_level = dci_inst->num_points_on_level;
    additional_info* root = dci_inst->root;

    if (num_levels == 0) {
        printf("No levels in DCI structure\n");
        return;
    }

    // Get the last level (finest level)
    int last_level = 0;
    int num_points_on_last_level = num_points_on_level[last_level];

    if (num_points_on_last_level == 0) {
        printf("No points on the finest level\n");
        return;
    }

    // Get all points on the last level
    additional_info** level_points = (additional_info**)malloc(sizeof(additional_info*) * num_points_on_last_level);

    // Traverse the tree to collect points on the last level
    int idx = num_levels - 1;
    int num_points_on_upper_levels = 1, num_points_on_cur_levels = 0;
    btree_p* cur_tree;
    btree_p_search_res s;
    additional_info*** all_level_points = (additional_info***)malloc(sizeof(additional_info**) * num_levels);

    while (idx >= 0) {
        num_points_on_cur_levels = num_points_on_level[idx];
        all_level_points[idx] = (additional_info**)malloc(sizeof(additional_info*) * num_points_on_cur_levels);
        int hh = 0;
        for (int j = 0; j < num_points_on_upper_levels; j++) {
            cur_tree = root->cell_indices;
            if (idx < num_levels - 1)
                cur_tree = all_level_points[idx + 1][j]->cell_indices;
            for (s = btree_p_first(cur_tree); !btree_p_is_end(cur_tree, s); s = btree_p_find_next(s)) {
                all_level_points[idx][hh++] = btree_p_valueof(s).info;
            }
        }
        num_points_on_upper_levels = num_points_on_cur_levels;
        idx -= 1;
    }

    // Copy points from the last level
    for (int i = 0; i < num_points_on_last_level; i++) {
        level_points[i] = all_level_points[0][i];
    }

    // Sort tokens by their id from small to large
    for (int i = 0; i < num_points_on_last_level - 1; i++) {
        for (int j = 0; j < num_points_on_last_level - 1 - i; j++) {
            if (level_points[j]->id > level_points[j + 1]->id) {
                additional_info* temp = level_points[j];
                level_points[j] = level_points[j + 1];
                level_points[j + 1] = temp;
            }
        }
    }

    // For each point P at the last level
    for (int p_idx = 0; p_idx < num_points_on_last_level; p_idx++) {
        additional_info* point_p = level_points[p_idx];

        // Check if this point is an anchor node (multiple of interval or last token)
        bool is_anchor = (interval > 0) && ((p_idx % interval == 0) || (p_idx == num_points_on_last_level - 1));

        if (is_anchor) {
            // For anchor nodes, set parent check and consistency to true, distances to 0
            parent_in_anchor_set[p_idx] = true;
            distance_ratios[p_idx] = 1.0f;
            parent_consistency[p_idx] = true;
            min_dist_to_anchor_parents[p_idx] = 0.0f;
            max_dist_closest_parent_to_children[p_idx] = 0.0f;
        } else {
            // For non-anchor points, find the closest anchor point and check against 2X anchor points
            int left_anchor_idx = (p_idx / interval) * interval;
            int right_anchor_idx = left_anchor_idx + interval;

            // Ensure we don't go out of bounds
            if (right_anchor_idx >= num_points_on_last_level) {
                right_anchor_idx = num_points_on_last_level - 1;
            }
            int dim = dci_inst->dim;

            // Collect the 2X anchor points (X on left, X on right)
            additional_info** anchor_points = (additional_info**)malloc(sizeof(additional_info*) * (2 * X));
            for (int k = 0; k < 2 * X; k++) {
                anchor_points[k] = NULL;
            }
            int num_anchor_points = 0;

            // Add left X anchor points
            for (int i = 0; i < X && (left_anchor_idx - i * interval) >= 0; i++) {
                int anchor_pos = left_anchor_idx - i * interval;
                if (anchor_pos >= 0) {
                    anchor_points[num_anchor_points++] = level_points[anchor_pos];
                }
            }

            // Add right X anchor points
            for (int i = 0; i < X && (right_anchor_idx + i * interval) < num_points_on_last_level; i++) {
                int anchor_pos = right_anchor_idx + i * interval;
                if (anchor_pos < num_points_on_last_level) {
                    anchor_points[num_anchor_points++] = level_points[anchor_pos];
                }
            }

            // Check if P's parent matches any of the 2X anchor points' parents (for parent_in_anchor_set)
            parent_in_anchor_set[p_idx] = false;
            if (point_p->parent_info != NULL) {
                for (int i = 0; i < num_anchor_points; i++) {
                    if (anchor_points[i] != NULL && anchor_points[i]->parent_info != NULL &&
                        point_p->parent_info->id == anchor_points[i]->parent_info->id) {
                        parent_in_anchor_set[p_idx] = true;
                        break;
                    }
                }
            }

            // Check if P's parent matches the closest anchor point's parent among the 2X anchors
            // and calculate the distance ratio
            parent_consistency[p_idx] = false;
            float dist_closest_anchor_to_its_parent = FLT_MAX;

            if (point_p->parent_info != NULL) {
                // Find the closest anchor among the 2X anchors
                float dist_to_closest_anchor = FLT_MAX;
                int closest_among_2X = -1;

                for (int i = 0; i < num_anchor_points; i++) {
                    if (anchor_points[i] != NULL && anchor_points[i]->data_loc != NULL && point_p->data_loc != NULL) {
                        float dist = transform_compute_dist(point_p->data_loc, anchor_points[i]->data_loc, dim,
                                                           dci_inst->max_sq_norm, dci_inst->sq_norm_list[point_p->id],
                                                           dci_inst->sq_norm_list[anchor_points[i]->id]);
                        if (dist < dist_to_closest_anchor) {
                            dist_to_closest_anchor = dist;
                            closest_among_2X = i;
                        }
                    }
                }

                // Check if parent matches the closest anchor's parent
                if (closest_among_2X >= 0 && anchor_points[closest_among_2X]->parent_info != NULL) {
                    parent_consistency[p_idx] = (point_p->parent_info->id == anchor_points[closest_among_2X]->parent_info->id);

                    // Calculate distance from closest anchor to its parent
                    if (anchor_points[closest_among_2X]->data_loc != NULL && anchor_points[closest_among_2X]->parent_info->data_loc != NULL) {
                        dist_closest_anchor_to_its_parent = transform_compute_dist(
                            anchor_points[closest_among_2X]->data_loc,
                            anchor_points[closest_among_2X]->parent_info->data_loc,
                            dim,
                            dci_inst->max_sq_norm,
                            dci_inst->sq_norm_list[anchor_points[closest_among_2X]->id],
                            dci_inst->sq_norm_list[anchor_points[closest_among_2X]->parent_info->id]
                        );
                    }
                }
            }

            // Calculate minimum distance to any parent in the anchor set
            float min_dist_to_any_anchor_parent = FLT_MAX;
            for (int i = 0; i < num_anchor_points; i++) {
                if (anchor_points[i] != NULL && anchor_points[i]->parent_info != NULL &&
                    anchor_points[i]->parent_info->data_loc != NULL && point_p->data_loc != NULL) {
                    float dist = transform_compute_dist(
                        point_p->data_loc,
                        anchor_points[i]->parent_info->data_loc,
                        dim,
                        dci_inst->max_sq_norm,
                        dci_inst->sq_norm_list[point_p->id],
                        dci_inst->sq_norm_list[anchor_points[i]->parent_info->id]
                    );
                    if (dist < min_dist_to_any_anchor_parent) {
                        min_dist_to_any_anchor_parent = dist;
                    }
                }
            }
            min_dist_to_anchor_parents[p_idx] = (min_dist_to_any_anchor_parent == FLT_MAX) ? 0.0f : min_dist_to_any_anchor_parent;

            // Calculate the ratio: min_dist_to_any_anchor_parent / points_toits_parent
            if (min_dist_to_any_anchor_parent != FLT_MAX) {
                distance_ratios[p_idx] = min_dist_to_any_anchor_parent / point_p->parent_dist;
            } else {
                distance_ratios[p_idx] = 1.0f; // Default ratio if division is not possible
            }

            // Find the anchor parent with minimum distance and calculate max distance to its children
            max_dist_closest_parent_to_children[p_idx] = 0.0f;

            // First, find which anchor parent has the minimum distance
            int closest_parent_idx = -1;
            for (int i = 0; i < num_anchor_points; i++) {
                if (anchor_points[i] != NULL && anchor_points[i]->parent_info != NULL &&
                    anchor_points[i]->parent_info->data_loc != NULL && point_p->data_loc != NULL) {
                    float dist = transform_compute_dist(
                        point_p->data_loc,
                        anchor_points[i]->parent_info->data_loc,
                        dim,
                        dci_inst->max_sq_norm,
                        dci_inst->sq_norm_list[point_p->id],
                        dci_inst->sq_norm_list[anchor_points[i]->parent_info->id]
                    );
                    if (dist == min_dist_to_any_anchor_parent) {
                        closest_parent_idx = i;
                        break;
                    }
                }
            }

            // Now find the maximum distance from this parent to all its children
            if (closest_parent_idx >= 0 && anchor_points[closest_parent_idx]->parent_info != NULL) {
                additional_info* closest_parent = anchor_points[closest_parent_idx]->parent_info;
                float max_dist_to_children = 0.0f;

                // Iterate through all children of this parent
                if (closest_parent->cell_indices != NULL) {
                    btree_p* parent_tree = closest_parent->cell_indices;
                    btree_p_search_res s;
                    for (s = btree_p_first(parent_tree); !btree_p_is_end(parent_tree, s); s = btree_p_find_next(s)) {
                        additional_info* child = btree_p_valueof(s).info;
                        if (child != NULL && child->data_loc != NULL && closest_parent->data_loc != NULL) {
                            float dist = transform_compute_dist(
                                closest_parent->data_loc,
                                child->data_loc,
                                dim,
                                dci_inst->max_sq_norm,
                                dci_inst->sq_norm_list[closest_parent->id],
                                dci_inst->sq_norm_list[child->id]
                            );
                            if (dist > max_dist_to_children) {
                                max_dist_to_children = dist;
                            }
                        }
                    }
                }
                max_dist_closest_parent_to_children[p_idx] = max_dist_to_children;
            }

            free(anchor_points);
        }
    }

    // Cleanup
    free(level_points);

    for (int i = 0; i < num_levels; i++) {
        free(all_level_points[i]);
    }
    free(all_level_points);
}

// Print the whole tree given the root
void print_tree(int num_levels, int* num_points_on_level, additional_info* root) {
    int j;
    int idx = num_levels - 1;
    int num_points_on_upper_levels = 1, num_points_on_cur_levels = 0;
    btree_p* cur_tree;
    btree_p_search_res s;
    additional_info*** level_points = (additional_info***)malloc(sizeof(additional_info**)*(num_levels));

    while (idx >= 0) {
        num_points_on_cur_levels = num_points_on_level[idx];
        level_points[idx] = (additional_info**)malloc(
            sizeof(additional_info*) * (num_points_on_cur_levels));
        int hh = 0;
        for (j = 0; j < num_points_on_upper_levels; j++) {
            cur_tree = root->cell_indices;
            if (idx < num_levels - 1)
                cur_tree = level_points[idx + 1][j]->cell_indices;
            for (s = btree_p_first(cur_tree); !btree_p_is_end(cur_tree, s); s = btree_p_find_next(s)) {
                level_points[idx][hh++] = btree_p_valueof(s).info;
            }
        }
        num_points_on_upper_levels = num_points_on_cur_levels;
        idx -= 1;
    }
    for (idx = num_levels - 1; idx >= 0; idx--) {
        num_points_on_cur_levels = num_points_on_level[idx];
        if (idx > 0) {
            for (j = 0; j < num_points_on_cur_levels; j++) {
                printf("%d(%d)  ", level_points[idx][j]->id, level_points[idx][j]->num_finest_level_points[0]);
            }
        }
        else {
            for (j = 0; j < num_points_on_cur_levels; j++) {
                printf("%d(1)  ", level_points[idx][j]->id);
            }
        }
        printf("\n");
    }
    printf("=============\n");

    idx = num_levels - 1;
    while (idx >= 0) {
        free(level_points[idx]);
        idx -= 1;
    }
    free(level_points);
}

// Print the number of children for each cell
void print_cell_num(dci* dci_inst) {
    int j;
    int num_levels = dci_inst->num_levels;
    int* num_points_on_level = dci_inst->num_points_on_level;
    additional_info* root = dci_inst->root;
    int idx = num_levels - 1;
    int num_points_on_upper_levels = 1, num_points_on_cur_levels = 0;
    btree_p* cur_tree;
    btree_p_search_res s;
    additional_info*** level_points = (additional_info***)malloc(sizeof(additional_info**)*(num_levels));

    while (idx >= 0) {
        num_points_on_cur_levels = num_points_on_level[idx];
        level_points[idx] = (additional_info**)malloc(
            sizeof(additional_info*) * (num_points_on_cur_levels));
        int hh = 0;
        for (j = 0; j < num_points_on_upper_levels; j++) {
            cur_tree = root->cell_indices;
            if (idx < num_levels - 1)
                cur_tree = level_points[idx + 1][j]->cell_indices;
            int temp_n = 0;
            for (s = btree_p_first(cur_tree); !btree_p_is_end(cur_tree, s); s = btree_p_find_next(s)) {
                level_points[idx][hh++] = btree_p_valueof(s).info;
                temp_n++;
            }
            printf("%d ", temp_n);
        }
        num_points_on_upper_levels = num_points_on_cur_levels;
        idx -= 1;
        printf("\n");
    }
    printf("*************\n");

    idx = num_levels - 1;
    while (idx >= 0) {
        free(level_points[idx]);
        idx -= 1;
    }
    free(level_points);
}

void print_num_points_on_level(dci* dci_inst) {
    int i;
    int num_levels = dci_inst->num_levels;
    int* num_points_on_level = dci_inst->num_points_on_level;
    for (i = num_levels - 1; i >= 0; i--) {
        printf("Level-%d: %d\n", i, num_points_on_level[i]);
    }
}

// Check the correctness of each node in DCI tree, and the cosistency with points_on_level
void check_dci(dci* dci_inst) {
    int i, j;
    int num_levels = dci_inst->num_levels;
    int* num_points_on_level = dci_inst->num_points_on_level;
    additional_info*** points_on_level = dci_inst->points_on_level;
    additional_info* root = dci_inst->root;
    int idx = num_levels - 1;
    int num_points_on_upper_levels = 1, num_points_on_cur_levels = 0;
    btree_p* cur_tree;
    btree_p_search_res s;
    additional_info*** level_points = (additional_info***)malloc(sizeof(additional_info**)*(num_levels));

    while (idx >= 0) {
        num_points_on_cur_levels = num_points_on_level[idx];
        level_points[idx] = (additional_info**)malloc(
            sizeof(additional_info*) * (num_points_on_cur_levels));
        int hh = 0;
        for (j = 0; j < num_points_on_upper_levels; j++) {
            cur_tree = root->cell_indices;
            if (idx < num_levels - 1) {
                cur_tree = level_points[idx + 1][j]->cell_indices;
            }
            for (s = btree_p_first(cur_tree); !btree_p_is_end(cur_tree, s); s = btree_p_find_next(s)) {
                level_points[idx][hh] = btree_p_valueof(s).info;
                // Check the correctness of each node
                if (idx < num_levels - 1)
                    assert(level_points[idx][hh]->parent_info == level_points[idx + 1][j]);
                else 
                    assert(level_points[idx][hh]->parent_info == root);
                hh++;
            }
        }
        assert(hh == num_points_on_cur_levels);
        num_points_on_upper_levels = num_points_on_cur_levels;
        idx -= 1;
    }
    for (idx = num_levels - 1; idx >= 0; idx--) {
        num_points_on_cur_levels = num_points_on_level[idx];
        for (i = 0; i < num_points_on_cur_levels; i++) {
            for (j = 0; j < num_points_on_cur_levels; j++) {
                if (points_on_level[idx][j] == level_points[idx][i]) {
                    break;
                }
            }
            assert(j < num_points_on_cur_levels);
        }
    }

    if (num_levels >= 2) {
        //  Check num_finest_level_points and num_finest_level_nodes
        num_points_on_cur_levels = num_points_on_level[1];
        for (j = 0; j < num_points_on_cur_levels; j++) {
            int temp_num_data = level_points[1][j]->cell_indices[0].num_data;
            assert(level_points[1][j]->num_finest_level_points[0] == temp_num_data + 1);
            assert(level_points[1][j]->num_finest_level_points[1] == temp_num_data);
            int temp_num_node = level_points[1][j]->cell_indices[0].num_leaf_nodes;
            assert(level_points[1][j]->num_finest_level_nodes[0] == temp_num_node + 1);
            assert(level_points[1][j]->num_finest_level_nodes[1] == temp_num_node);
        }
        for (i = 2; i < dci_inst->num_levels; i++) {
            num_points_on_cur_levels = num_points_on_level[i];
            for (j = 0; j < num_points_on_cur_levels; j++) {
                int num_finest_level_points[i + 1];
                int num_finest_level_nodes[i + 1];
                for (int l = i; l >= 0; l--) {
                    num_finest_level_points[l] = 0;
                    num_finest_level_nodes[l] = 0;
                }
                num_finest_level_points[i] = level_points[i][j]->cell_indices[0].num_data;
                num_finest_level_points[0] = 1;
                num_finest_level_nodes[i] = level_points[i][j]->cell_indices[0].num_leaf_nodes;
                num_finest_level_nodes[0] = 1;
                cur_tree = level_points[i][j]->cell_indices;
                assert(cur_tree != NULL);
                btree_p_search_res s;

                for (s = btree_p_first(cur_tree); !btree_p_is_end(cur_tree, s);
                    s = btree_p_find_next(s)) {
                    for (int l = i - 1; l >= 0; l--) {
                        num_finest_level_points[l] += s.n->slot_data[s.slot].info->num_finest_level_points[l];
                        num_finest_level_nodes[l] += s.n->slot_data[s.slot].info->num_finest_level_nodes[l];
                    }
                }
                num_finest_level_nodes[0] -= num_finest_level_points[i];
                num_finest_level_nodes[0] += num_finest_level_nodes[i];
                for (int l = i; l >= 0; l--) {
                    assert(level_points[i][j]->num_finest_level_points[l] == num_finest_level_points[l]);
                    assert(level_points[i][j]->num_finest_level_nodes[l] == num_finest_level_nodes[l]);
                }
            }
        }
    }

    idx = num_levels - 1;
    while (idx >= 0) {
        free(level_points[idx]);
        idx -= 1;
    }
    free(level_points);
}

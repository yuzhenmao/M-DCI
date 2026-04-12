#include <Python.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <iostream>
#include "hashtable_wrapper.h"
#include "dci.h"
#include "debug.h"
#include "btree_i.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <any>
namespace py = pybind11;


typedef struct py_dci {
    dci* dci_inst;           // The DCI instance
    HashTable* hashtable;    // A pointer to a hash table
    btree_i* cached_tree;    // A pointer to a cached tree
} py_dci;

typedef struct py_dci_list {
    py_dci* dci_inst_list;  // Array of py_dci instances
    int num_inst;           // Number of instances in the list
} py_dci_list;


class PyDCI {
public:
    PyDCI(
        int dim,
        int num_comp_indices,
        int num_simp_indices,
        float promotion_prob,
        float promotion_prob_subseq,
        int max_volume,
        bool transform,
        int num_inst,
        int parallel_level,
        bool debug,
        const torch::Tensor& proj_vec_t
    ) {
        if (promotion_prob_subseq == 0.0f) {
            promotion_prob_subseq = promotion_prob;
        }

        float* proj_vec = nullptr;
        if (proj_vec_t.defined() && proj_vec_t.numel() > 0) {
            TORCH_CHECK(proj_vec_t.device().is_cpu(), "proj_vec must be a CPU tensor");
            TORCH_CHECK(proj_vec_t.scalar_type() == torch::kFloat32, "proj_vec must be float32");
            TORCH_CHECK(proj_vec_t.is_contiguous(), "proj_vec must be contiguous");

            proj_vec = proj_vec_t.data_ptr<float>();
        }

        dci_list.num_inst = num_inst;
        dci_list.dci_inst_list = new py_dci[num_inst];
        py_dci* pdci_inst_list = dci_list.dci_inst_list;

        for (int i = 0; i < num_inst; i++) {
            pdci_inst_list[i].dci_inst = new dci;
            dci_init(
                pdci_inst_list[i].dci_inst,
                dim, num_comp_indices, num_simp_indices,
                promotion_prob, promotion_prob_subseq,
                max_volume, transform, parallel_level, debug,
                proj_vec
            );
            pdci_inst_list[i].hashtable = new HashTable(1, max_volume);  // match legacy
            // match legacy
            pdci_inst_list[i].cached_tree = nullptr;
        }
    }

    ~PyDCI() {
        py_dci* pdci_inst_list = dci_list.dci_inst_list;
        if (!pdci_inst_list) return;

        for (int i = 0; i < dci_list.num_inst; i++) {
            dci_free(pdci_inst_list[i].dci_inst);
            delete pdci_inst_list[i].dci_inst;
            pdci_inst_list[i].dci_inst = nullptr;
            pdci_inst_list[i].hashtable->clear();
            delete pdci_inst_list[i].hashtable;
            pdci_inst_list[i].hashtable = nullptr;
            if (pdci_inst_list[i].cached_tree) {
                btree_i_clear(pdci_inst_list[i].cached_tree);
                free(pdci_inst_list[i].cached_tree);     // IMPORTANT: only if it was malloc'ed
                pdci_inst_list[i].cached_tree = nullptr;
            }
        }

        delete[] dci_list.dci_inst_list;
        dci_list.dci_inst_list = nullptr;
        dci_list.num_inst = 0;
    }

    torch::Tensor get_num_levels() const {
        int num_inst = dci_list.num_inst;
        std::vector<int> num_levels(num_inst);
        py_dci* pdci_inst_list = dci_list.dci_inst_list;
        for (int i = 0; i < num_inst; i++) {
            num_levels[i] = pdci_inst_list[i].dci_inst->num_levels;
        }

        return torch::from_blob(num_levels.data(), {num_inst}, torch::dtype(torch::kInt32)).clone();
    }

    torch::Tensor get_num_points() const {
        int num_inst = dci_list.num_inst;
        std::vector<int> num_points(num_inst);
        py_dci* pdci_inst_list = dci_list.dci_inst_list;
        for (int i = 0; i < num_inst; i++) {
            num_points[i] = pdci_inst_list[i].dci_inst->num_points;
        }
        return torch::from_blob(num_points.data(), {num_inst}, torch::dtype(torch::kInt32)).clone();
    }

    std::tuple<torch::Tensor, torch::Tensor> get_token2node() const {
        int num_inst = dci_list.num_inst;
        int num_points = dci_list.dci_inst_list[0].dci_inst->num_points;
        int* token2node_index = new int[num_inst * num_points];
        int* token2node_offset = new int[num_inst * num_points];
        py_dci* pdci_inst_list = dci_list.dci_inst_list;
        for (int i = 0; i < num_inst; i++) {
            for (int j = 0; j < num_points; j++) {
                token2node_index[i * num_points + j] = pdci_inst_list[i].dci_inst->token2nodeIndex[j];
                token2node_offset[i * num_points + j] = pdci_inst_list[i].dci_inst->token2nodeOffset[j];
            }
        }

        torch::Tensor index_tensor = torch::from_blob(token2node_index, {num_inst, num_points}, torch::dtype(torch::kInt32)).clone();
        torch::Tensor offset_tensor = torch::from_blob(token2node_offset, {num_inst, num_points}, torch::dtype(torch::kInt32)).clone();

        delete[] token2node_index;
        delete[] token2node_offset;
        
        return std::make_tuple(index_tensor, offset_tensor);
    }

    void dci_print(int idx) const {
        print_dci(dci_list.dci_inst_list[idx].dci_inst);
    }

    void cell_num_print(int idx) const {
        print_cell_num(dci_list.dci_inst_list[idx].dci_inst);
    }

    void num_points_on_level_print(int idx) const {
        print_num_points_on_level(dci_list.dci_inst_list[idx].dci_inst);
    }

    void dci_check() const {
        for (int i = 0; i < dci_list.num_inst; i++) {
            check_dci(dci_list.dci_inst_list[i].dci_inst);
        }
    }

    std::tuple<torch::Tensor>
    add_query_torch(
        const torch::Tensor& key,                 // [num_inst, max_num_points, dim], float32, CPU
        const torch::Tensor& query,               // [num_inst, max_num_points, dim], float32, CPU (only if do_query)
        const torch::Tensor& value,               // [num_inst, max_num_points, dim], float32, CPU
        const torch::Tensor& mask,                // [num_inst, max_num_points], bool, CPU
        int num_levels,
        int num_neighbours,
        unsigned char blind,
        int c_num_to_visit,
        int q_num_to_visit,
        int c_num_to_retrieve,
        int q_num_to_retrieve,
        float c_prop_to_visit,
        float q_prop_to_visit,
        float c_prop_to_retrieve,
        float q_prop_to_retrieve,
        int c_field_of_view,
        int q_field_of_view,
        int num_comp_indices,
        int num_simp_indices,
        int64_t max_num_points,                   // per inst
        bool transform,                           // (kept for parity; dci_init owns this)
        int parallel_level,
        const torch::Tensor& attention_mask,      // [max_num_points] or empty, int32, CPU. If empty -> treat as causal
        bool random,
        bool do_query,
        bool track,
        bool update_addr,
        torch::Tensor& changed_page_list,
        const torch::Tensor& data_proj_all,       // optional: [num_inst, max_num_points, num_indices] float32 or empty
        int ratio,
        int interval,
        int X,
        float anchor_threshold
    ) {
        py::gil_scoped_release release_gil;

        TORCH_CHECK(ratio == 1, "add_query_torch currently does not support group query (ratio must be 1).");

        TORCH_CHECK(key.device().is_cpu(), "key must be a CPU tensor");
        TORCH_CHECK(value.device().is_cpu(), "value must be a CPU tensor");
        TORCH_CHECK(mask.device().is_cpu(), "mask must be a CPU tensor");
        if (do_query) TORCH_CHECK(query.device().is_cpu(), "query must be a CPU tensor when do_query=true");
        if (attention_mask.defined() && attention_mask.numel() > 0) {
            TORCH_CHECK(attention_mask.device().is_cpu(), "attention_mask must be a CPU tensor");
        }
        if (data_proj_all.defined() && data_proj_all.numel() > 0) {
            TORCH_CHECK(data_proj_all.device().is_cpu(), "data_proj_all must be a CPU tensor");
        }

        TORCH_CHECK(key.scalar_type() == torch::kFloat32, "key must be float32");
        TORCH_CHECK(value.scalar_type() == torch::kFloat32, "value must be float32");
        TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");
        if (do_query) TORCH_CHECK(query.scalar_type() == torch::kFloat32, "query must be float32");
        if (attention_mask.defined() && attention_mask.numel() > 0) {
            TORCH_CHECK(attention_mask.scalar_type() == torch::kInt32, "attention_mask must be int32");
        }
        if (data_proj_all.defined() && data_proj_all.numel() > 0) {
            TORCH_CHECK(data_proj_all.scalar_type() == torch::kFloat32, "data_proj_all must be float32");
        }

        TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
        TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
        TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
        if (do_query) TORCH_CHECK(query.is_contiguous(), "query must be contiguous when do_query=true");
        if (attention_mask.defined() && attention_mask.numel() > 0) TORCH_CHECK(attention_mask.is_contiguous(), "attention_mask must be contiguous");
        if (data_proj_all.defined() && data_proj_all.numel() > 0) TORCH_CHECK(data_proj_all.is_contiguous(), "data_proj_all must be contiguous");

        TORCH_CHECK(key.dim() == 3, "key must have shape [num_inst, max_num_points, dim]");
        TORCH_CHECK(value.dim() == 3, "value must have shape [num_inst, max_num_points, dim]");
        TORCH_CHECK(mask.dim() == 2, "mask must have shape [num_inst, max_num_points]");
        if (do_query) TORCH_CHECK(query.dim() == 3, "query must have shape [num_inst, max_num_points, dim] when do_query=true");

        const int64_t num_inst = dci_list.num_inst;
        TORCH_CHECK(key.size(0) == num_inst, "key.size(0) must equal num_inst");
        TORCH_CHECK(value.size(0) == num_inst, "value.size(0) must equal num_inst");
        TORCH_CHECK(mask.size(0) == num_inst, "mask.size(0) must equal num_inst");
        if (do_query) TORCH_CHECK(query.size(0) == num_inst, "query.size(0) must equal num_inst");

        TORCH_CHECK(key.size(1) == max_num_points, "key.size(1) must equal max_num_points");
        TORCH_CHECK(value.size(1) == max_num_points, "value.size(1) must equal max_num_points");
        TORCH_CHECK(mask.size(1) == max_num_points, "mask.size(1) must equal max_num_points");
        if (do_query) TORCH_CHECK(query.size(1) == max_num_points, "query.size(1) must equal max_num_points");

        const int64_t dim = key.size(2);
        TORCH_CHECK(value.size(2) == dim, "value dim mismatch");
        if (do_query) TORCH_CHECK(query.size(2) == dim, "query dim mismatch");

        const int num_indices = num_comp_indices * num_simp_indices;

        const float* key_ptr   = key.data_ptr<float>();
        const float* value_ptr = value.data_ptr<float>();
        const float* query_ptr = do_query ? query.data_ptr<float>() : nullptr;
        const bool*  mask_ptr  = mask.data_ptr<bool>();

        // Build a local attention_mask buffer (do NOT mutate input tensor)
        std::vector<int> attn_local(static_cast<size_t>(max_num_points));
        
        TORCH_CHECK(attention_mask.numel() == max_num_points, "attention_mask must have numel == max_num_points");
        const int32_t* attn_in = attention_mask.data_ptr<int32_t>();
        for (int64_t i = 0; i < max_num_points; ++i) attn_local[static_cast<size_t>(i)] = static_cast<int>(attn_in[i]);

        // Allocate outputs
        // matches your NumPy behavior: store (idx, dist_or_aux?) pairs => 2 ints per neighbor
        auto nearest_idx = torch::zeros(
            {num_inst, max_num_points, num_neighbours, 2},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
        );
        auto num_returned_t = torch::zeros(
            {num_inst, max_num_points},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
        );

        int32_t* nearest_out = nearest_idx.data_ptr<int32_t>();
        int32_t* num_ret_out = num_returned_t.data_ptr<int32_t>();

        bool* changed_out = nullptr;
        int old_page_num = 0;
        if (track) {
            // Use first instance leaf node count as "old_page_num" like your original logic did with array dim
            TORCH_CHECK(changed_page_list.defined());
            TORCH_CHECK(changed_page_list.device().is_cpu(), "changed_page_list must be a CPU tensor");
            TORCH_CHECK(changed_page_list.scalar_type() == torch::kBool, "changed_page_list must be bool");
            TORCH_CHECK(changed_page_list.is_contiguous(), "changed_page_list must be contiguous");
            TORCH_CHECK(changed_page_list.dim() == 2, "changed_page_list must have shape [num_inst, old_page_num]");
            TORCH_CHECK(changed_page_list.size(0) == num_inst, "changed_page_list.size(0) must equal num_inst");
            old_page_num = changed_page_list.size(1);
            changed_out = changed_page_list.data_ptr<bool>();
        }

        // Count valid points globally (your original code did this on mask[0:max_num_points] without idx offset)
        // To preserve behavior, we use instance 0’s mask for the global clamp.
        int64_t num_points_valid0 = 0;
        for (int64_t i = 0; i < max_num_points; ++i) {
            if (mask_ptr[i]) ++num_points_valid0;
        }
        for (int64_t i = 0; i < max_num_points; ++i) {
            if (attn_local[static_cast<size_t>(i)] > num_points_valid0) attn_local[static_cast<size_t>(i)] = static_cast<int>(num_points_valid0);
        }

        dci_query_config construction_query_config;
        construction_query_config.blind = blind;
        construction_query_config.num_to_visit = c_num_to_visit;
        construction_query_config.num_to_retrieve = c_num_to_retrieve;
        construction_query_config.prop_to_visit = c_prop_to_visit;
        construction_query_config.prop_to_retrieve = c_prop_to_retrieve;
        construction_query_config.field_of_view = c_field_of_view;
        construction_query_config.target_level = 0;

        dci_query_config query_config;
        if (do_query) {
            query_config.blind = blind;
            query_config.num_to_visit = q_num_to_visit;
            query_config.num_to_retrieve = q_num_to_retrieve;
            query_config.prop_to_visit = q_prop_to_visit;
            query_config.prop_to_retrieve = q_prop_to_retrieve;
            query_config.field_of_view = q_field_of_view;
            query_config.target_level = 0;
        }

        // Thread setup (kept from your NumPy version)
        static int original_max_threads = 0;
        if (original_max_threads == 0) original_max_threads = omp_get_max_threads();

        int inner_threads = 1;
        int inner_inner_threads = 1;
        if (parallel_level >= 2) {
            omp_set_nested(1);
            int max_levels = (parallel_level >= 3) ? 3 : 2;
            omp_set_max_active_levels(max_levels);

            int total_threads = original_max_threads;
            int outer_threads = (num_inst < total_threads) ? (int)num_inst : total_threads / 2;
            if (outer_threads < 1) outer_threads = 1;
            inner_threads = total_threads / outer_threads;
            if (inner_threads < 1) inner_threads = 1;

            if (parallel_level >= 3 && inner_threads > 1) {
                inner_inner_threads = (inner_threads > 2) ? 2 : 1;
                inner_threads = (inner_threads > 2) ? inner_threads / 2 : inner_threads;
                if (inner_threads < 1) inner_threads = 1;
            }

            omp_set_num_threads(outer_threads);
        }

        // Optional precomputed projections
        const float* data_proj_all_ptr = nullptr;
        bool has_precomputed = false;
        if (data_proj_all.defined() && data_proj_all.numel() > 0) {
            TORCH_CHECK(data_proj_all.dim() == 3, "data_proj_all must be [num_inst, max_num_points, num_indices]");
            TORCH_CHECK(data_proj_all.size(0) == num_inst, "data_proj_all.size(0) mismatch");
            TORCH_CHECK(data_proj_all.size(1) == max_num_points, "data_proj_all.size(1) mismatch");
            TORCH_CHECK(data_proj_all.size(2) == num_indices, "data_proj_all.size(2) mismatch");
            data_proj_all_ptr = data_proj_all.data_ptr<float>();
            has_precomputed = true;
        }

    #pragma omp parallel for if(parallel_level >= 1)
        for (int inst = 0; inst < (int)num_inst; ++inst) {
            dci* dci_inst = dci_list.dci_inst_list[inst].dci_inst;
            dci_inst->parallel_level = parallel_level;
            dci_inst->inner_threads = inner_threads;
            dci_inst->inner_inner_threads = inner_inner_threads;
            dci_inst->update_addr = update_addr;

            const int prev_num_leaf_nodes = dci_inst->num_leaf_nodes;

            // Base pointers for this instance
            const float* key_i   = key_ptr   + (int64_t)inst * max_num_points * dim;
            const float* value_i = value_ptr + (int64_t)inst * max_num_points * dim;
            const float* query_i = do_query ? (query_ptr + (int64_t)inst * max_num_points * dim) : nullptr;
            const bool*  mask_i  = mask_ptr  + (int64_t)inst * max_num_points;

            // Projections: either precomputed view, or allocate inside data_projection
            float* data_proj = nullptr;
            bool pre_computed = false;
            if (has_precomputed) {
                pre_computed = true;
                data_proj = const_cast<float*>(data_proj_all_ptr + (int64_t)inst * max_num_points * num_indices);
            }

            data_projection(
                num_indices, dci_inst, (int)dim, (long long)max_num_points,
                const_cast<float*>(key_i), &data_proj, const_cast<bool*>(mask_i),
                pre_computed
            );

            // nearest neighbours temp buffer (DCI allocates per query point like your original)
            int** nearest_tmp = (int**)malloc(sizeof(int*) * (size_t)max_num_points);

            if (max_num_points > 1 && attn_local[0] > 1) {
                if (do_query) {
                    // num_returned output pointer for this instance (shape [max_num_points])
                    int* num_ret_i = (int*)malloc(sizeof(int) * (size_t)max_num_points);

                    dci_add_query(
                        dci_inst, (int)dim, attn_local[0],
                        const_cast<float*>(key_i), const_cast<float*>(value_i), num_levels,
                        construction_query_config, NULL, 0,
                        data_proj, const_cast<bool*>(mask_i),
                        attn_local[0], const_cast<float*>(query_i),
                        num_neighbours, query_config,
                        const_cast<bool*>(mask_i),
                        nearest_tmp,
                        NULL,
                        num_ret_i,
                        random,
                        interval, X, anchor_threshold
                    );

                    // write outputs
                    for (int i = 0; i < attn_local[0]; ++i) {
                        if (!mask_i[i]) continue;

                        num_ret_out[(int64_t)inst * max_num_points + i] = (int32_t)num_ret_i[i];

                        // store 2 ints per neighbor (like your NumPy flattened *2)
                        const int limit2 = num_ret_i[i] * 2;
                        const int cap2   = num_neighbours * 2;
                        const int write2 = (limit2 < cap2) ? limit2 : cap2;

                        int32_t* out_base = nearest_out
                            + ((int64_t)inst * max_num_points * num_neighbours * 2)
                            + ((int64_t)i * num_neighbours * 2);

                        for (int j = 0; j < write2; ++j) out_base[j] = (int32_t)nearest_tmp[i][j];

                        // free per-point allocation
                        free(nearest_tmp[i]);
                    }

                    free(num_ret_i);
                } else {
                    dci_add(
                        dci_inst, (int)dim, attn_local[0],
                        const_cast<float*>(key_i), const_cast<float*>(value_i), num_levels,
                        construction_query_config, NULL, 0,
                        data_proj, const_cast<bool*>(mask_i),
                        random, interval, X, anchor_threshold
                    );
                }
            } else {
                // point-by-point path (same as your original else-branch)
                for (int64_t i = 0; i < max_num_points; ++i) {
                    if (!mask_i[i]) continue;

                    int target_level;
                    float promotion_prob = dci_inst->promotion_prob;

                    if (random) {
                        target_level = 0;
                        while (1) {
                            if (target_level > 0) promotion_prob = dci_inst->promotion_prob_subseq;
                            if (drand48() > promotion_prob) break;
                            target_level++;
                        }
                    } else {
                        if (dci_inst->num_levels == 0) {
                            target_level = 0;
                            dci_inst->next_target_level = 0;
                        } else {
                            target_level = dci_inst->next_target_level;
                            if (target_level > 0) promotion_prob = dci_inst->promotion_prob_subseq;

                            if (target_level == dci_inst->num_levels) {
                                dci_inst->next_target_level = 0;
                            } else if (target_level == dci_inst->num_levels - 1) {
                                int promo = (int)ceil(1 / promotion_prob);
                                if (dci_inst->num_points_on_level[target_level] + 1 == promo)
                                    dci_inst->next_target_level = target_level + 1;
                                else
                                    dci_inst->next_target_level = 0;
                            } else {
                                if (dci_inst->num_points_on_level[target_level] + 1 >=
                                    dci_inst->num_points_on_level[target_level + 1] / promotion_prob)
                                    dci_inst->next_target_level = target_level + 1;
                                else
                                    dci_inst->next_target_level = 0;
                            }
                        }
                    }

                    if (dci_inst->num_points < num_neighbours || !do_query) {
                        dci_add(
                            dci_inst, (int)dim, 1,
                            const_cast<float*>(key_i + i * dim),
                            const_cast<float*>(value_i + i * dim),
                            dci_inst->num_levels, construction_query_config,
                            &(dci_inst->next_point_id), target_level,
                            data_proj + i * num_indices,
                            const_cast<bool*>(mask_i + i),
                            random, 0, 1, anchor_threshold
                        );

                        if (do_query) {
                            // vanilla fallback: neighbors are [0..i]
                            const int cap = (int)std::min<int64_t>(i + 1, num_neighbours);
                            num_ret_out[(int64_t)inst * max_num_points + i] = (int32_t)cap;

                            int32_t* out_base = nearest_out
                                + ((int64_t)inst * max_num_points * num_neighbours * 2)
                                + ((int64_t)i * num_neighbours * 2);
                            for (int j = 0; j < cap; ++j) {
                                out_base[j * 2 + 0] = j;
                                out_base[j * 2 + 1] = 0;
                            }
                        }
                    } else {
                        // query path: dci_add_query for single point
                        int num_ret_i = 0;
                        int** nearest_single = nearest_tmp; // reuse buffer

                        dci_add_query(
                            dci_inst, (int)dim, 1,
                            const_cast<float*>(key_i + i * dim),
                            const_cast<float*>(value_i + i * dim),
                            dci_inst->num_levels,
                            construction_query_config, &(dci_inst->next_point_id), target_level,
                            data_proj + i * num_indices, const_cast<bool*>(mask_i + i),
                            1, const_cast<float*>(query_i + i * dim),
                            num_neighbours, query_config,
                            const_cast<bool*>(mask_i + i),
                            nearest_single,
                            NULL,
                            &num_ret_i,
                            random,
                            0, 1, anchor_threshold
                        );

                        num_ret_out[(int64_t)inst * max_num_points + i] = (int32_t)num_ret_i;

                        int32_t* out_base = nearest_out
                            + ((int64_t)inst * max_num_points * num_neighbours * 2)
                            + ((int64_t)i * num_neighbours * 2);

                        const int limit2 = num_ret_i * 2;
                        const int cap2   = num_neighbours * 2;
                        const int write2 = (limit2 < cap2) ? limit2 : cap2;

                        for (int j = 0; j < write2; ++j) out_base[j] = (int32_t)nearest_single[0][j];
                        free(nearest_single[0]);
                    }
                }
            }

            if (track) {
                bool* page_status = dci_inst->page_status;
                // Preserve your original semantics: only fill up to prev_num_leaf_nodes, reset status
                for (int i = 0; i < prev_num_leaf_nodes; ++i) {
                    if (page_status[i]) {
                        changed_out[inst * old_page_num + i] = 1;
                        page_status[i] = false;
                    }
                    else {
                        changed_out[inst * old_page_num + i] = 0;
                    }
                }
            }

            free(nearest_tmp);

            if (!pre_computed) {
                free(data_proj);
            }
        }

        if (parallel_level >= 2) {
            omp_set_num_threads(original_max_threads);
        }

        return nearest_idx;
    }

    std::tuple<torch::Tensor, torch::Tensor>
    query_torch(
        const torch::Tensor& query,    // [num_query_head, num_query, dim], float32, CPU
        const torch::Tensor& mask,     // [num_query_head, num_query], bool, CPU
        int num_neighbours,
        bool blind,
        int q_num_to_visit,
        int q_num_to_retrieve,
        float q_prop_to_visit,
        float q_prop_to_retrieve,
        int q_field_of_view,
        int parallel_level,
        int ratio
    ) {
        py::gil_scoped_release release_gil;

        TORCH_CHECK(query.device().is_cpu(), "query must be a CPU tensor");
        TORCH_CHECK(mask.device().is_cpu(), "mask must be a CPU tensor");
        TORCH_CHECK(query.scalar_type() == torch::kFloat32, "query must be float32");
        TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");
        TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
        TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
        TORCH_CHECK(query.dim() == 3, "query must be [num_query_head, num_query, dim]");
        TORCH_CHECK(mask.dim() == 2, "mask must be [num_query_head, num_query]");

        const int64_t num_query_head = query.size(0);
        const int64_t num_query      = query.size(1);
        const int64_t dim            = query.size(2);

        TORCH_CHECK(mask.size(0) == num_query_head, "mask.size(0) mismatch");
        TORCH_CHECK(mask.size(1) == num_query, "mask.size(1) mismatch");

        const int num_inst = dci_list.num_inst;
        TORCH_CHECK(num_query_head == (int64_t)num_inst * ratio,
                    "num_query_head must equal num_inst * ratio");

        dci_query_config query_config;
        query_config.blind = blind;
        query_config.num_to_visit = q_num_to_visit;
        query_config.num_to_retrieve = q_num_to_retrieve;
        query_config.prop_to_visit = q_prop_to_visit;
        query_config.prop_to_retrieve = q_prop_to_retrieve;
        query_config.field_of_view = q_field_of_view;
        query_config.target_level = 0;

        // NumPy-equivalent outputs:
        // nearest_neighbour_idx: flat length = num_query * num_neighbours * num_query_head * 2
        auto nearest_neighbour_idx = torch::zeros(
            {num_query * num_neighbours * num_query_head * 2},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
        );

        // num_returned: flat length = num_query * num_query_head
        auto num_returned_t = torch::zeros(
            {num_query * num_query_head},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
        );

        const float*  query_ptr = query.data_ptr<float>();
        const bool*   mask_ptr  = mask.data_ptr<bool>();
        int32_t*      nn_out    = nearest_neighbour_idx.data_ptr<int32_t>();
        int32_t*      nr_out    = num_returned_t.data_ptr<int32_t>();

        // Temporary: array of pointers sized [num_query_head * num_query]
        // We also zero-init for safety.
        int64_t total_ptrs = num_query_head * num_query;
        int** nearest_neighbours = (int**)calloc((size_t)total_ptrs, sizeof(int*));
        TORCH_CHECK(nearest_neighbours != nullptr, "calloc failed");

    #pragma omp parallel for if(parallel_level >= 1)
        for (int qh = 0; qh < (int)num_query_head; ++qh) {
            int dci_idx = qh / ratio;
            dci* dci_inst = dci_list.dci_inst_list[dci_idx].dci_inst;
            dci_inst->parallel_level = parallel_level;

            TORCH_CHECK((int)dim == dci_inst->dim, "query dim does not match dci_inst->dim");

            // pointers for this head
            const float* query_i = query_ptr + (int64_t)qh * num_query * dim;
            const bool*  mask_i  = mask_ptr  + (int64_t)qh * num_query;

            int** nearest_temp = &(nearest_neighbours[(int64_t)qh * num_query]);
            int32_t* num_ret_i = nr_out + (int64_t)qh * num_query;

            // If your dci_query expects int*, this is safe only if sizeof(int)==4.
            static_assert(sizeof(int) == 4, "dci_query assumes 32-bit int");
            dci_query(
                dci_inst,
                (int)dim,
                (int)num_query,
                const_cast<float*>(query_i),
                num_neighbours,
                query_config,
                const_cast<bool*>(mask_i),
                nearest_temp,
                NULL,
                reinterpret_cast<int*>(num_ret_i)
            );

            // Copy layout identical to NumPy:
            // nn_out base for head qh: qh * num_query * num_neighbours * 2
            int64_t head_base = (int64_t)qh * num_query * num_neighbours * 2;

            for (int i = 0; i < (int)num_query; ++i) {
                if (!mask_i[i]) continue;

                const int num_ret = (int)num_ret_i[i];
                const int limit2  = num_ret * 2;
                const int cap2    = num_neighbours * 2;
                const int write2  = (limit2 < cap2) ? limit2 : cap2;

                int64_t out_base = head_base + (int64_t)i * num_neighbours * 2;

                // nearest_temp[i] should contain 2* num_returned ints
                for (int j = 0; j < write2; ++j) {
                    nn_out[out_base + j] = (int32_t)nearest_temp[i][j];
                }

                // Free allocation for this query point
                free(nearest_temp[i]);
                nearest_temp[i] = nullptr;
            }

            // Safety: if dci_query ever allocates for masked-out points, clean them too
            for (int i = 0; i < (int)num_query; ++i) {
                if (nearest_temp[i] != nullptr) {
                    free(nearest_temp[i]);
                    nearest_temp[i] = nullptr;
                }
            }
        }

        free(nearest_neighbours);

        return std::make_tuple(nearest_neighbour_idx, num_returned_t);
    }


    py::array_t<float> get_proj_vec() {
        // Calculate the shape of the projection vector
        dci* dci_inst = dci_list.dci_inst_list[0].dci_inst;
        py::ssize_t num_comp_simp_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;
        py::ssize_t dim = dci_inst->dim;
        
        // Define the shape
        std::vector<py::ssize_t> shape = { num_comp_simp_indices, dim };
        
        // Create the NumPy array
        auto py_proj_vec = py::array_t<float>(shape);

        // Map the underlying data to the projection vector
        std::memcpy(py_proj_vec.mutable_data(), dci_inst->proj_vec, sizeof(float) * num_comp_simp_indices * dim);
        
        return py_proj_vec;
    }

    void reset() {
        py_dci* pdci_inst_list = dci_list.dci_inst_list;
        for (int i = 0; i < dci_list.num_inst; i++) {
            dci_reset(pdci_inst_list[i].dci_inst);

            int expected_num_entries = pdci_inst_list[i].hashtable->getSize();
            pdci_inst_list[i].hashtable->clear();
            delete pdci_inst_list[i].hashtable;
            pdci_inst_list[i].hashtable = new HashTable(1, expected_num_entries);
            btree_i_clear(pdci_inst_list[i].cached_tree);
            delete pdci_inst_list[i].cached_tree;
            pdci_inst_list[i].cached_tree = new btree_i;
            btree_i_init(pdci_inst_list[i].cached_tree);
        }
    }
        
    private:
        py_dci_list dci_list;  // List of py_dci instances
};



PYBIND11_MODULE(dci, m) {
    py::class_<PyDCI>(m, "PyDCI")
        .def(py::init<int, int, int, float, float, int, bool, int, int, bool, torch::Tensor>(),
            py::arg("dim"), 
            py::arg("num_comp_indices"), 
            py::arg("num_simp_indices"), 
            py::arg("promotion_prob"),
            py::arg("promotion_prob_subseq"),
            py::arg("max_volume"),
            py::arg("transform"),
            py::arg("num_inst"),
            py::arg("parallel_level"),
            py::arg("debug"),
            py::arg("proj_vec_t")
        )
        .def("get_num_points", &PyDCI::get_num_points)
        .def("add_query_torch", &PyDCI::add_query_torch,
            py::arg("key"),
            py::arg("query"),
            py::arg("value"),
            py::arg("mask"),
            py::arg("num_levels"),
            py::arg("num_neighbours"),
            py::arg("blind"),
            py::arg("c_num_to_visit"),
            py::arg("q_num_to_visit"),
            py::arg("c_num_to_retrieve"),
            py::arg("q_num_to_retrieve"),
            py::arg("c_prop_to_visit"),
            py::arg("q_prop_to_visit"),
            py::arg("c_prop_to_retrieve"),
            py::arg("q_prop_to_retrieve"),
            py::arg("c_field_of_view"),
            py::arg("q_field_of_view"),
            py::arg("num_comp_indices"),
            py::arg("num_simp_indices"),
            py::arg("max_num_points"),
            py::arg("transform"),
            py::arg("parallel_level"),
            py::arg("attention_mask") = torch::Tensor(),
            py::arg("random"),
            py::arg("do_query"),
            py::arg("track"),
            py::arg("update_addr"),
            py::arg("changed_page_list"),
            py::arg("data_proj_all") = torch::Tensor(),
            py::arg("ratio") = 1,
            py::arg("interval") = 0,
            py::arg("X") = 1,
            py::arg("anchor_threshold") = 0.0f
        )
        .def("query_torch", &PyDCI::query_torch,
            py::arg("query"),
            py::arg("mask"),
            py::arg("num_neighbours"),
            py::arg("blind"),
            py::arg("q_num_to_visit"),
            py::arg("q_num_to_retrieve"),
            py::arg("q_prop_to_visit"),
            py::arg("q_prop_to_retrieve"),
            py::arg("q_field_of_view"),
            py::arg("parallel_level"),
            py::arg("ratio") = 1
        )
        .def("get_num_levels", &PyDCI::get_num_levels)
        .def("get_token2node", &PyDCI::get_token2node)
        .def("get_proj_vec", &PyDCI::get_proj_vec)
        .def("dci_print", &PyDCI::dci_print, py::arg("idx"))
        .def("cell_num_print", &PyDCI::cell_num_print, py::arg("idx"))
        .def("num_points_on_level_print", &PyDCI::num_points_on_level_print, py::arg("idx"))
        .def("dci_check", &PyDCI::dci_check)
        .def("reset", &PyDCI::reset);
}
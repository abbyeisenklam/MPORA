#include "xgb_utils_userspace.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

typedef struct {
    XGBEnsemble ens;
} XGBHandleImpl;

typedef struct {
    XGBScalingFactors scales;
} XGBScalingHandleImpl;

static int read_u32(FILE* f, uint32_t* v) {
    return fread(v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}
static int read_i32(FILE* f, int32_t* v) {
    return fread(v, sizeof(int32_t), 1, f) == 1 ? 0 : -1;
}
static int read_u16(FILE* f, uint16_t* v) {
    return fread(v, sizeof(uint16_t), 1, f) == 1 ? 0 : -1;
}
static int read_i16(FILE* f, int16_t* v) {
    return fread(v, sizeof(int16_t), 1, f) == 1 ? 0 : -1;
}
static int read_u8(FILE* f, uint8_t* v) {
    return fread(v, sizeof(uint8_t), 1, f) == 1 ? 0 : -1;
}
static int read_i64(FILE* f, int64_t* v) {
    return fread(v, sizeof(int64_t), 1, f) == 1 ? 0 : -1;
}

xgb_handle_t xgb_load_ensemble(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    XGBHandleImpl* h = (XGBHandleImpl*)calloc(1, sizeof(XGBHandleImpl));
    if (!h) { fclose(f); return NULL; }

    // Binary layout:
    // int32 input_dim
    // int32 base_q
    // uint32 num_trees
    int32_t input_dim = 0;
    int32_t base_q = 0;
    uint32_t num_trees = 0;
    if (read_i32(f, &input_dim) || read_i32(f, &base_q) || read_u32(f, &num_trees)) {
        fclose(f); free(h); return NULL;
    }
    h->ens.input_dim = input_dim;
    h->ens.base_q = base_q;
    h->ens.num_trees = num_trees;
    h->ens.trees = (XGBTree*)calloc(num_trees, sizeof(XGBTree));
    if (!h->ens.trees) { fclose(f); free(h); return NULL; }

    for (uint32_t ti = 0; ti < num_trees; ++ti) {
        uint32_t num_nodes = 0;
        if (read_u32(f, &num_nodes)) { fclose(f); xgb_free_ensemble(h); return NULL; }
        h->ens.trees[ti].num_nodes = num_nodes;
        uint32_t root_idx = 0;
        if (read_u32(f, &root_idx)) { fclose(f); xgb_free_ensemble(h); return NULL; }
        h->ens.trees[ti].root_idx = root_idx;
        h->ens.trees[ti].nodes = (XGBNode*)calloc(num_nodes, sizeof(XGBNode));
        if (!h->ens.trees[ti].nodes) { fclose(f); xgb_free_ensemble(h); return NULL; }

        int use_u16_child = (num_nodes <= 65535);
        int use_i16_fidx = (input_dim <= 32767);

        for (uint32_t ni = 0; ni < num_nodes; ++ni) {
            uint8_t is_leaf = 0;
            if (read_u8(f, &is_leaf)) { fclose(f); xgb_free_ensemble(h); return NULL; }
            XGBNode* n = &h->ens.trees[ti].nodes[ni];
            n->is_leaf = is_leaf;
            if (is_leaf) {
                int32_t leaf_q = 0;
                if (read_i32(f, &leaf_q)) { fclose(f); xgb_free_ensemble(h); return NULL; }
                n->leaf_q = leaf_q;
                n->fidx = -1;
                n->thr_q = 0;
                n->yes = 0;
                n->no = 0;
            } else {
                int32_t fidx32 = 0;
                if (use_i16_fidx) {
                    int16_t fidx16 = 0;
                    if (read_i16(f, &fidx16)) { fclose(f); xgb_free_ensemble(h); return NULL; }
                    fidx32 = (int32_t)fidx16;
                } else {
                    if (read_i32(f, &fidx32)) { fclose(f); xgb_free_ensemble(h); return NULL; }
                }
                int32_t thr_q = 0;
                if (read_i32(f, &thr_q)) { fclose(f); xgb_free_ensemble(h); return NULL; }
                int32_t yes_i = 0, no_i = 0;
                if (use_u16_child) {
                    uint16_t y16 = 0, n16 = 0;
                    if (read_u16(f, &y16) || read_u16(f, &n16)) { fclose(f); xgb_free_ensemble(h); return NULL; }
                    yes_i = (int32_t)y16;
                    no_i = (int32_t)n16;
                } else {
                    uint32_t y32 = 0, n32 = 0;
                    if (read_u32(f, &y32) || read_u32(f, &n32)) { fclose(f); xgb_free_ensemble(h); return NULL; }
                    yes_i = (int32_t)y32;
                    no_i = (int32_t)n32;
                }
                n->fidx = fidx32;
                n->thr_q = thr_q;
                n->yes = yes_i;
                n->no = no_i;
                n->leaf_q = 0;
            }
        }
    }

    fclose(f);
    return (xgb_handle_t)h;
}

void xgb_free_ensemble(xgb_handle_t handle) {
    if (!handle) return;
    XGBHandleImpl* h = (XGBHandleImpl*)handle;
    if (h->ens.trees) {
        for (uint32_t ti = 0; ti < h->ens.num_trees; ++ti) {
            free(h->ens.trees[ti].nodes);
        }
        free(h->ens.trees);
    }
    free(h);
}

static inline int32_t predict_tree(const XGBTree* t, const int32_t* x_q) {
    int32_t nid = (int32_t)t->root_idx;
    while (1) {
        const XGBNode* n = &t->nodes[nid];
        if (n->is_leaf) {
            return n->leaf_q;
        }
        int32_t fidx = n->fidx;
        int32_t xv = x_q[fidx]; // assume valid index; caller ensures input_dim
        nid = (xv < n->thr_q) ? n->yes : n->no;
    }
}

int xgb_predict_sample(xgb_handle_t handle, const int32_t* x_q, int32_t* out_q) {
    if (!handle || !x_q || !out_q) return -1;
    XGBHandleImpl* h = (XGBHandleImpl*)handle;
    int64_t acc = h->ens.base_q;
    for (uint32_t ti = 0; ti < h->ens.num_trees; ++ti) {
        acc += predict_tree(&h->ens.trees[ti], x_q);
    }
    // Saturate to int32
    if (acc > INT32_MAX) acc = INT32_MAX;
    if (acc < INT32_MIN) acc = INT32_MIN;
    *out_q = (int32_t)acc;
    return 0;
}

int xgb_predict_batch(xgb_handle_t handle,
                      const int32_t* inputs_q,
                      int input_stride,
                      int batch_size,
                      int32_t* outputs_q) {
    if (!handle || !inputs_q || !outputs_q) return -1;
    XGBHandleImpl* h = (XGBHandleImpl*)handle;
    for (int i = 0; i < batch_size; ++i) {
        const int32_t* xi = inputs_q + i * input_stride;
        int32_t yi;
        if (xgb_predict_sample(handle, xi, &yi) != 0) return -1;
        outputs_q[i] = yi;
    }
    return 0;
}

xgb_scaling_handle_t xgb_load_scaling_factors(const char* path) {
    if (!path) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    XGBScalingHandleImpl* h = (XGBScalingHandleImpl*)calloc(1, sizeof(XGBScalingHandleImpl));
    if (!h) {
        fclose(f);
        return NULL;
    }

    char magic[8];
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic)) {
        fclose(f);
        free(h);
        return NULL;
    }
    if (memcmp(magic, "XGBSCL01", 8) != 0) {
        fclose(f);
        free(h);
        return NULL;
    }

    uint32_t version = 0;
    uint32_t frac_bits = 0;
    uint32_t num_inputs = 0;
    uint32_t num_outputs = 0;
    if (read_u32(f, &version) || read_u32(f, &frac_bits) || read_u32(f, &num_inputs) || read_u32(f, &num_outputs)) {
        fclose(f);
        free(h);
        return NULL;
    }

    if (version != 1) {
        fclose(f);
        free(h);
        return NULL;
    }
    if (num_inputs != XGB_SCALE_NUM_INPUTS || num_outputs != XGB_SCALE_NUM_OUTPUTS) {
        fclose(f);
        free(h);
        return NULL;
    }

    h->scales.frac_bits = (int32_t)frac_bits;
    h->scales.num_inputs = num_inputs;
    h->scales.num_outputs = num_outputs;

    for (uint32_t i = 0; i < num_inputs; ++i) {
        if (read_i64(f, &h->scales.input_scales_q[i])) {
            fclose(f);
            free(h);
            return NULL;
        }
    }
    for (uint32_t i = 0; i < num_outputs; ++i) {
        if (read_i64(f, &h->scales.output_scales_q[i])) {
            fclose(f);
            free(h);
            return NULL;
        }
    }

    fclose(f);
    return (xgb_scaling_handle_t)h;
}

void xgb_free_scaling_factors(xgb_scaling_handle_t handle) {
    if (!handle) return;
    XGBScalingHandleImpl* h = (XGBScalingHandleImpl*)handle;
    free(h);
}

int xgb_get_input_scale_q(xgb_scaling_handle_t handle, int feature_index, int64_t* out_q) {
    if (!handle || !out_q) return -1;
    if (feature_index < 0 || feature_index >= (int)XGB_SCALE_NUM_INPUTS) return -1;
    XGBScalingHandleImpl* h = (XGBScalingHandleImpl*)handle;
    *out_q = h->scales.input_scales_q[feature_index];
    return 0;
}

int xgb_get_output_scale_q(xgb_scaling_handle_t handle, int output_index, int64_t* out_q) {
    if (!handle || !out_q) return -1;
    if (output_index < 0 || output_index >= (int)XGB_SCALE_NUM_OUTPUTS) return -1;
    XGBScalingHandleImpl* h = (XGBScalingHandleImpl*)handle;
    *out_q = h->scales.output_scales_q[output_index];
    return 0;
}

int xgb_get_scaling_frac_bits(xgb_scaling_handle_t handle, int32_t* out_frac_bits) {
    if (!handle || !out_frac_bits) return -1;
    XGBScalingHandleImpl* h = (XGBScalingHandleImpl*)handle;
    *out_frac_bits = h->scales.frac_bits;
    return 0;
}

typedef struct {
    const XGBModelBank* bank;
    const XGBJobState* jobs;
    int num_jobs;
    int64_t pred_rem_ns[XGB_ALLOC_MAX_JOBS][XGB_ALLOC_MAX_BUDGET - XGB_ALLOC_MIN_BUDGET + 1];
    int64_t pred_next_insn[XGB_ALLOC_MAX_JOBS][XGB_ALLOC_MAX_BUDGET - XGB_ALLOC_MIN_BUDGET + 1];
    int current_alloc[XGB_ALLOC_MAX_JOBS];
    int best_alloc[XGB_ALLOC_MAX_JOBS];
    int found_feasible;
    int64_t best_total_next_insn;
    int64_t best_total_tardiness_ns;
} AllocSearchCtx;

static const char* kTaskNames[XGB_TASK_COUNT] = {
    "mcf",
    "omnetpp",
    "xz",
    "xalancbmk",
    "exchange2",
    "x264",
};

static const char* kCanonicalInputFeatures[XGB_SCALE_NUM_INPUTS] = {
    "insn_sum",
    "prev_budget",
    "next_budget",
    "prev_insn",
    "inp_feat1",
    "inp_feat2",
    "inp_feat3",
    "inp_feat4",
    "inp_feat5",
    "inp_feat6",
    "prev_insn_1",
    "prev_insn_2",
    "prev_insn_3",
    "prev_insn_4",
    "prev_insn_5",
    "prev_budget_1",
    "prev_budget_2",
    "prev_budget_3",
    "prev_budget_4",
    "prev_budget_5",
};

static int32_t clamp_i64_to_i32(int64_t x) {
    if (x > (int64_t)INT32_MAX) return INT32_MAX;
    if (x < (int64_t)INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

static int64_t sat_add_i64(int64_t a, int64_t b) {
    if (b > 0 && a > INT64_MAX - b) return INT64_MAX;
    if (b < 0 && a < INT64_MIN - b) return INT64_MIN;
    return a + b;
}

static int64_t clamp_i128_to_i64(__int128 x) {
    if (x > (__int128)INT64_MAX) return INT64_MAX;
    if (x < (__int128)INT64_MIN) return INT64_MIN;
    return (int64_t)x;
}

static int64_t div_round_i128(__int128 num, int64_t den) {
    if (den <= 0) return 0;
    __int128 abs_num = (num < 0) ? -num : num;
    __int128 q = (abs_num + (den / 2)) / den;
    if (num < 0) q = -q;
    return clamp_i128_to_i64(q);
}

static int64_t normalize_raw_to_q16(int64_t raw_val, int64_t scale_q, int32_t scale_frac_bits) {
    if (scale_q <= 0) return 0;
    int32_t shift = 16 + scale_frac_bits;
    if (shift < 0 || shift > 60) return 0;
    __int128 num = ((__int128)raw_val) << (uint32_t)shift;
    return div_round_i128(num, scale_q);
}

static int64_t rescale_pred_q16_to_raw(int32_t pred_q16, int64_t scale_q, int32_t scale_frac_bits) {
    if (scale_q <= 0) return 0;
    int32_t shift = 16 + scale_frac_bits;
    if (shift < 0 || shift > 60) return 0;
    int64_t den = (int64_t)1 << (uint32_t)shift;
    __int128 num = (__int128)pred_q16 * (__int128)scale_q;
    return div_round_i128(num, den);
}

static int64_t rescale_rem_time_q16_to_ns(int32_t pred_q16, int64_t rem_scale_q, int32_t scale_frac_bits) {
    if (rem_scale_q <= 0) return 0;
    int32_t shift = 16 + scale_frac_bits;
    if (shift < 0 || shift > 60) return 0;
    int64_t den = (int64_t)1 << (uint32_t)shift;
    __int128 num = (__int128)pred_q16 * (__int128)rem_scale_q * 1000000000LL;
    int64_t ns = div_round_i128(num, den);
    if (ns < 0) return 0;
    return ns;
}

static int file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static int canonical_feature_index(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < XGB_SCALE_NUM_INPUTS; ++i) {
        if (strcmp(name, kCanonicalInputFeatures[i]) == 0) return i;
    }
    return -1;
}

static int parse_model_feature_order(
    const char* meta_path,
    int32_t* out_indices,
    int max_indices,
    int* out_count
) {
    if (!meta_path || !out_indices || !out_count || max_indices <= 0) return -1;

    FILE* f = fopen(meta_path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long sz = ftell(f);
    if (sz <= 0 || sz > 65536) {
        fclose(f);
        return -1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }

    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return -1;
    }
    size_t nread = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (nread != (size_t)sz) {
        free(buf);
        return -1;
    }
    buf[sz] = '\0';

    char* p = strstr(buf, "\"feature_order\"");
    if (!p) {
        free(buf);
        return -1;
    }
    p = strchr(p, '[');
    if (!p) {
        free(buf);
        return -1;
    }
    p++;

    int count = 0;
    while (*p && *p != ']') {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
        if (*p == ']') break;
        if (*p != '"') {
            free(buf);
            return -1;
        }
        p++;
        char* start = p;
        while (*p && *p != '"') p++;
        if (*p != '"') {
            free(buf);
            return -1;
        }

        char saved = *p;
        *p = '\0';
        int idx = canonical_feature_index(start);
        *p = saved;
        if (idx < 0) {
            free(buf);
            return -1;
        }
        if (count >= max_indices) {
            free(buf);
            return -1;
        }
        out_indices[count++] = (int32_t)idx;
        p++;
    }

    free(buf);
    if (count <= 0) return -1;
    *out_count = count;
    return 0;
}

static int load_task_bundle(XGBTaskModelBundle* bundle, const char* models_root_dir, const char* task_name) {
    char path0[512];
    char path1[512];
    char path_scaling[512];
    char path0_meta[512];
    char path1_meta[512];
    int n0 = snprintf(path0, sizeof(path0), "%s/%s/xgb_fp_target_0.bin", models_root_dir, task_name);
    int n1 = snprintf(path1, sizeof(path1), "%s/%s/xgb_fp_target_1.bin", models_root_dir, task_name);
    int ns = snprintf(path_scaling, sizeof(path_scaling), "%s/%s/scaling_factors.bin", models_root_dir, task_name);
    int n0m = snprintf(path0_meta, sizeof(path0_meta), "%s/%s/xgb_fp_target_0.bin.meta.json", models_root_dir, task_name);
    int nm = snprintf(path1_meta, sizeof(path1_meta), "%s/%s/xgb_fp_target_1.bin.meta.json", models_root_dir, task_name);
    if (n0 <= 0 || n1 <= 0 || ns <= 0 || n0m <= 0 || nm <= 0 || n0 >= (int)sizeof(path0) || n1 >= (int)sizeof(path1) || ns >= (int)sizeof(path_scaling) || n0m >= (int)sizeof(path0_meta) || nm >= (int)sizeof(path1_meta)) {
        return -1;
    }

    if (!file_exists(path0) || !file_exists(path1) || !file_exists(path_scaling)) {
        return 1;
    }

    memset(bundle, 0, sizeof(*bundle));
    bundle->target0_rem_time = xgb_load_ensemble(path0);
    if (!bundle->target0_rem_time) return -1;

    bundle->target1_next_insn = xgb_load_ensemble(path1);
    if (!bundle->target1_next_insn) return -1;

    bundle->scaling = xgb_load_scaling_factors(path_scaling);
    if (!bundle->scaling) return -1;

    if (xgb_get_scaling_frac_bits(bundle->scaling, &bundle->scaling_frac_bits) != 0) return -1;
    for (int i = 0; i < XGB_SCALE_NUM_INPUTS; ++i) {
        if (xgb_get_input_scale_q(bundle->scaling, i, &bundle->input_scales_q[i]) != 0) return -1;
        bundle->model0_to_canonical_idx[i] = -1;
        bundle->model1_to_canonical_idx[i] = -1;
    }
    for (int i = 0; i < XGB_SCALE_NUM_OUTPUTS; ++i) {
        if (xgb_get_output_scale_q(bundle->scaling, i, &bundle->output_scales_q[i]) != 0) return -1;
    }

    int parsed_count0 = 0;
    if (file_exists(path0_meta) &&
        parse_model_feature_order(path0_meta, bundle->model0_to_canonical_idx, XGB_SCALE_NUM_INPUTS, &parsed_count0) == 0) {
        bundle->model0_input_dim = parsed_count0;
    } else {
        int c = 0;
        for (int i = 0; i < XGB_SCALE_NUM_INPUTS; ++i) {
            if (bundle->input_scales_q[i] > 0) {
                bundle->model0_to_canonical_idx[c++] = i;
            }
        }
        bundle->model0_input_dim = c;
    }

    int parsed_count1 = 0;
    if (file_exists(path1_meta) &&
        parse_model_feature_order(path1_meta, bundle->model1_to_canonical_idx, XGB_SCALE_NUM_INPUTS, &parsed_count1) == 0) {
        bundle->model1_input_dim = parsed_count1;
    } else {
        int c = 0;
        for (int i = 0; i < XGB_SCALE_NUM_INPUTS; ++i) {
            if (bundle->input_scales_q[i] > 0) {
                bundle->model1_to_canonical_idx[c++] = i;
            }
        }
        bundle->model1_input_dim = c;
    }

    if (bundle->model0_input_dim <= 0 || bundle->model0_input_dim > XGB_SCALE_NUM_INPUTS) return -1;
    if (bundle->model1_input_dim <= 0 || bundle->model1_input_dim > XGB_SCALE_NUM_INPUTS) return -1;
    return 0;
}

const char* xgb_task_to_string(XGBTaskName task) {
    if ((int)task < 0 || task >= XGB_TASK_COUNT) return "invalid";
    return kTaskNames[(int)task];
}

XGBTaskName xgb_task_from_string(const char* task_name) {
    if (!task_name) return XGB_TASK_INVALID;
    for (int i = 0; i < XGB_TASK_COUNT; ++i) {
        if (strcmp(task_name, kTaskNames[i]) == 0) {
            return (XGBTaskName)i;
        }
    }
    return XGB_TASK_INVALID;
}

int xgb_model_bank_init(XGBModelBank* bank, const char* models_root_dir) {
    if (!bank || !models_root_dir) return -1;
    memset(bank, 0, sizeof(*bank));
    int loaded_count = 0;
    for (int task = 0; task < XGB_TASK_COUNT; ++task) {
        int rc = load_task_bundle(&bank->task_models[task], models_root_dir, kTaskNames[task]);
        if (rc == 0) {
            loaded_count++;
            continue;
        }
        if (rc < 0) {
            xgb_model_bank_free(bank);
            return -1;
        }
    }
    if (loaded_count == 0) {
        xgb_model_bank_free(bank);
        return -1;
    }
    return 0;
}

void xgb_model_bank_free(XGBModelBank* bank) {
    if (!bank) return;
    for (int task = 0; task < XGB_TASK_COUNT; ++task) {
        XGBTaskModelBundle* b = &bank->task_models[task];
        if (b->target0_rem_time) xgb_free_ensemble(b->target0_rem_time);
        if (b->target1_next_insn) xgb_free_ensemble(b->target1_next_insn);
        if (b->scaling) xgb_free_scaling_factors(b->scaling);
        memset(b, 0, sizeof(*b));
    }
}

static int prepare_budget_predictions(AllocSearchCtx* ctx) {
    const int next_budget_canonical_idx = 2;
    const int budget_count = XGB_ALLOC_MAX_BUDGET - XGB_ALLOC_MIN_BUDGET + 1;
    for (int j = 0; j < ctx->num_jobs; ++j) {
        XGBTaskName task = ctx->jobs[j].task;
        if ((int)task < 0 || task >= XGB_TASK_COUNT) return -1;

        const XGBTaskModelBundle* bundle = &ctx->bank->task_models[task];
        if (!bundle->target0_rem_time || !bundle->target1_next_insn || !bundle->scaling) return -1;

        int32_t x_q0[XGB_SCALE_NUM_INPUTS];
        int32_t x_q1[XGB_SCALE_NUM_INPUTS];
        int32_t rem_q = 0;
        int32_t insn_q = 0;

        for (int bidx = 0; bidx < budget_count; ++bidx) {
            int budget = XGB_ALLOC_MIN_BUDGET + bidx;
            for (int f = 0; f < XGB_SCALE_NUM_INPUTS; ++f) {
                x_q0[f] = 0;
                x_q1[f] = 0;
            }

            int feature_count0 = bundle->model0_input_dim;
            if (feature_count0 > XGB_SCALE_NUM_INPUTS) feature_count0 = XGB_SCALE_NUM_INPUTS;
            for (int mf = 0; mf < feature_count0; ++mf) {
                int canonical_idx = bundle->model0_to_canonical_idx[mf];
                if (canonical_idx < 0 || canonical_idx >= XGB_SCALE_NUM_INPUTS) {
                    x_q0[mf] = 0;
                    continue;
                }

                int64_t scale_q = bundle->input_scales_q[canonical_idx];
                if (scale_q <= 0) {
                    x_q0[mf] = 0;
                    continue;
                }
                int64_t raw_val = ctx->jobs[j].input_features[canonical_idx];
                if (canonical_idx == next_budget_canonical_idx) {
                    raw_val = budget;
                }
                int64_t norm_q16 = normalize_raw_to_q16(raw_val, scale_q, bundle->scaling_frac_bits);
                x_q0[mf] = clamp_i64_to_i32(norm_q16);
            }

            int feature_count1 = bundle->model1_input_dim;
            if (feature_count1 > XGB_SCALE_NUM_INPUTS) feature_count1 = XGB_SCALE_NUM_INPUTS;
            for (int mf = 0; mf < feature_count1; ++mf) {
                int canonical_idx = bundle->model1_to_canonical_idx[mf];
                if (canonical_idx < 0 || canonical_idx >= XGB_SCALE_NUM_INPUTS) {
                    x_q1[mf] = 0;
                    continue;
                }

                int64_t scale_q = bundle->input_scales_q[canonical_idx];
                if (scale_q <= 0) {
                    x_q1[mf] = 0;
                    continue;
                }
                int64_t raw_val = ctx->jobs[j].input_features[canonical_idx];
                if (canonical_idx == next_budget_canonical_idx) {
                    raw_val = budget;
                }
                int64_t norm_q16 = normalize_raw_to_q16(raw_val, scale_q, bundle->scaling_frac_bits);
                x_q1[mf] = clamp_i64_to_i32(norm_q16);
            }

            if (xgb_predict_sample(bundle->target0_rem_time, x_q0, &rem_q) != 0) return -1;
            if (xgb_predict_sample(bundle->target1_next_insn, x_q1, &insn_q) != 0) return -1;

            int64_t rem_ns = rescale_rem_time_q16_to_ns(rem_q, bundle->output_scales_q[0], bundle->scaling_frac_bits);
            int64_t next_insn = rescale_pred_q16_to_raw(insn_q, bundle->output_scales_q[1], bundle->scaling_frac_bits);
            if (next_insn < 0) next_insn = 0;

            ctx->pred_rem_ns[j][bidx] = rem_ns;
            ctx->pred_next_insn[j][bidx] = next_insn;
        }
    }
    return 0;
}

static void evaluate_current_assignment(AllocSearchCtx* ctx) {
    int64_t total_next_insn = 0;
    int64_t total_tardiness_ns = 0;
    for (int j = 0; j < ctx->num_jobs; ++j) {
        int bidx = ctx->current_alloc[j] - XGB_ALLOC_MIN_BUDGET;
        int64_t pred_ns = ctx->pred_rem_ns[j][bidx];
        int64_t pred_insn = ctx->pred_next_insn[j][bidx];
        total_next_insn = sat_add_i64(total_next_insn, pred_insn);
        if (pred_ns > ctx->jobs[j].deadline_ns) {
            int64_t tard = pred_ns - ctx->jobs[j].deadline_ns;
            total_tardiness_ns = sat_add_i64(total_tardiness_ns, tard);
        }
    }

    int feasible = (total_tardiness_ns == 0);
    if (ctx->found_feasible) {
        if (!feasible) return;
        if (total_next_insn > ctx->best_total_next_insn) {
            ctx->best_total_next_insn = total_next_insn;
            ctx->best_total_tardiness_ns = 0;
            for (int i = 0; i < ctx->num_jobs; ++i) ctx->best_alloc[i] = ctx->current_alloc[i];
        }
        return;
    }

    if (feasible) {
        ctx->found_feasible = 1;
        ctx->best_total_next_insn = total_next_insn;
        ctx->best_total_tardiness_ns = 0;
        for (int i = 0; i < ctx->num_jobs; ++i) ctx->best_alloc[i] = ctx->current_alloc[i];
        return;
    }

    if (total_tardiness_ns < ctx->best_total_tardiness_ns ||
        (total_tardiness_ns == ctx->best_total_tardiness_ns && total_next_insn > ctx->best_total_next_insn)) {
        ctx->best_total_tardiness_ns = total_tardiness_ns;
        ctx->best_total_next_insn = total_next_insn;
        for (int i = 0; i < ctx->num_jobs; ++i) ctx->best_alloc[i] = ctx->current_alloc[i];
    }
}

static void dfs_budget_assignments(AllocSearchCtx* ctx, int job_idx, int used_budget) {
    if (job_idx == ctx->num_jobs) {
        if (used_budget != XGB_ALLOC_MAX_BUDGET) {
            return;
        }
        evaluate_current_assignment(ctx);
        return;
    }

    int remaining_jobs = ctx->num_jobs - (job_idx + 1);
    int min_remaining = remaining_jobs * XGB_ALLOC_MIN_BUDGET;
    int max_budget_for_job = XGB_ALLOC_MAX_BUDGET - (used_budget + min_remaining);
    if (max_budget_for_job > XGB_ALLOC_MAX_BUDGET) {
        max_budget_for_job = XGB_ALLOC_MAX_BUDGET;
    }
    if (max_budget_for_job < XGB_ALLOC_MIN_BUDGET) {
        return;
    }

    for (int b = XGB_ALLOC_MIN_BUDGET; b <= max_budget_for_job; ++b) {
        ctx->current_alloc[job_idx] = b;
        dfs_budget_assignments(ctx, job_idx + 1, used_budget + b);
    }
}

int allocate_resources(
    const XGBModelBank* bank,
    const XGBJobState* jobs,
    int num_jobs,
    int* out_allocations,
    int* out_feasible,
    int64_t* out_total_next_insn,
    int64_t* out_total_tardiness_ns
) {
    if (!bank || !jobs || !out_allocations) return -1;
    if (num_jobs < 0 || num_jobs > XGB_ALLOC_MAX_JOBS) return -1;

    for (int i = 0; i < XGB_ALLOC_MAX_JOBS; ++i) out_allocations[i] = XGB_ALLOC_MIN_BUDGET;
    if (num_jobs == 0) {
        if (out_feasible) *out_feasible = 1;
        if (out_total_next_insn) *out_total_next_insn = 0;
        if (out_total_tardiness_ns) *out_total_tardiness_ns = 0;
        return 0;
    }

    if (num_jobs * XGB_ALLOC_MIN_BUDGET > XGB_ALLOC_MAX_BUDGET) return -1;

    AllocSearchCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.bank = bank;
    ctx.jobs = jobs;
    ctx.num_jobs = num_jobs;
    ctx.found_feasible = 0;
    ctx.best_total_next_insn = -1;
    ctx.best_total_tardiness_ns = INT64_MAX;

    if (prepare_budget_predictions(&ctx) != 0) return -1;

    dfs_budget_assignments(&ctx, 0, 0);

    if (ctx.best_total_next_insn < 0) return -1;

    for (int i = 0; i < num_jobs; ++i) {
        out_allocations[i] = ctx.best_alloc[i];
    }
    if (out_feasible) *out_feasible = ctx.found_feasible ? 1 : 0;
    if (out_total_next_insn) *out_total_next_insn = ctx.best_total_next_insn;
    if (out_total_tardiness_ns) {
        if (ctx.best_total_tardiness_ns <= 0) {
            *out_total_tardiness_ns = 0;
        } else {
            *out_total_tardiness_ns = ctx.best_total_tardiness_ns;
        }
    }
    return 0;
}

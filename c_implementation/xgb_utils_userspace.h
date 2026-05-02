#ifndef XGB_UTILS_USERSPACE_H
#define XGB_UTILS_USERSPACE_H

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t is_leaf;   // 1 if leaf, 0 otherwise
    int32_t fidx;      // feature index (Q16.16 expects int32 input)
    int32_t thr_q;     // threshold in Q16.16
    int32_t yes;       // child index if x < thr
    int32_t no;        // child index if x >= thr
    int32_t leaf_q;    // leaf value in Q16.16
} XGBNode;

typedef struct {
    uint32_t num_nodes;
    uint32_t root_idx;  // index of root node in nodes[]
    XGBNode* nodes;    // array of nodes indexed 0..num_nodes-1; root is 0
} XGBTree;

typedef struct {
    int32_t base_q;        // base_score in Q16.16
    int32_t input_dim;     // number of input features expected
    uint32_t num_trees;
    XGBTree* trees;        // array of trees
} XGBEnsemble;

enum {
    XGB_SCALE_NUM_INPUTS = 20,
    XGB_SCALE_NUM_OUTPUTS = 2,
};

typedef struct {
    int32_t frac_bits;
    uint32_t num_inputs;
    uint32_t num_outputs;
    int64_t input_scales_q[XGB_SCALE_NUM_INPUTS];
    int64_t output_scales_q[XGB_SCALE_NUM_OUTPUTS];
} XGBScalingFactors;

// Opaque handle for dynamic linking
typedef void* xgb_handle_t;
typedef void* xgb_scaling_handle_t;

enum {
    XGB_ALLOC_MAX_JOBS = 4,
    XGB_ALLOC_MIN_BUDGET = 2,
    XGB_ALLOC_MAX_BUDGET = 20,
};

typedef enum {
    XGB_TASK_MCF = 0,
    XGB_TASK_OMNETPP,
    XGB_TASK_XZ,
    XGB_TASK_XALANCBMK,
    XGB_TASK_EXCHANGE2,
    XGB_TASK_X264,
    XGB_TASK_COUNT,
    XGB_TASK_INVALID = -1,
} XGBTaskName;

typedef struct {
    xgb_handle_t target0_rem_time;
    xgb_handle_t target1_next_insn;
    xgb_scaling_handle_t scaling;
    int32_t scaling_frac_bits;
    int64_t input_scales_q[XGB_SCALE_NUM_INPUTS];
    int64_t output_scales_q[XGB_SCALE_NUM_OUTPUTS];
    int32_t model0_input_dim;
    int32_t model1_input_dim;
    int32_t model0_to_canonical_idx[XGB_SCALE_NUM_INPUTS];
    int32_t model1_to_canonical_idx[XGB_SCALE_NUM_INPUTS];
} XGBTaskModelBundle;

typedef struct {
    XGBTaskModelBundle task_models[XGB_TASK_COUNT];
} XGBModelBank;

typedef struct {
    XGBTaskName task;
    int64_t deadline_ns;
    int64_t input_features[XGB_SCALE_NUM_INPUTS];
} XGBJobState;

// Load an ensemble from a binary file produced by Python exporter.
// Returns NULL on failure.
xgb_handle_t xgb_load_ensemble(const char* path);

// Free resources for an ensemble.
void xgb_free_ensemble(xgb_handle_t handle);

// Predict a single sample (inputs in Q16.16), returns 0 on success.
int xgb_predict_sample(xgb_handle_t handle, const int32_t* x_q, int32_t* out_q);

// Predict a batch of samples laid out AoS: [batch][input_dim] in Q16.16.
// output is [batch] in Q16.16. Returns 0 on success.
int xgb_predict_batch(xgb_handle_t handle,
                      const int32_t* inputs_q,
                      int input_stride,
                      int batch_size,
                      int32_t* outputs_q);

// Load fixed-point scaling factors binary produced by export_xgb_scaling_factors.py.
// Returns NULL on failure.
xgb_scaling_handle_t xgb_load_scaling_factors(const char* path);

// Free resources for scaling factors handle.
void xgb_free_scaling_factors(xgb_scaling_handle_t handle);

// Fetch one input scale (Q value); returns 0 on success.
int xgb_get_input_scale_q(xgb_scaling_handle_t handle, int feature_index, int64_t* out_q);

// Fetch one output scale (Q value); output index 0=rem_time, 1=next_insn. Returns 0 on success.
int xgb_get_output_scale_q(xgb_scaling_handle_t handle, int output_index, int64_t* out_q);

// Get the fractional bits used for Q representation (e.g., 16 for Q48.16).
int xgb_get_scaling_frac_bits(xgb_scaling_handle_t handle, int32_t* out_frac_bits);

const char* xgb_task_to_string(XGBTaskName task);

XGBTaskName xgb_task_from_string(const char* task_name);

int xgb_model_bank_init(XGBModelBank* bank, const char* models_root_dir);

void xgb_model_bank_free(XGBModelBank* bank);

int allocate_resources(
    const XGBModelBank* bank,
    const XGBJobState* jobs,
    int num_jobs,
    int* out_allocations,
    int* out_feasible,
    int64_t* out_total_next_insn,
    int64_t* out_total_tardiness_ns
);

#ifdef __cplusplus
}
#endif

#endif // XGB_UTILS_USERSPACE_H

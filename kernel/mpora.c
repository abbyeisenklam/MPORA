// SPDX-License-Identifier: GPL-2.0

#include <linux/kernel.h>
#include <linux/errno.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/math64.h>
#include <linux/limits.h>
#include <linux/ktime.h>
#include <asm/fpu/api.h>

#include <mpora/ml.h>
#include <mpora/ml_upload_ext.h>

#define MPORA_ML_TOTAL_PARTITIONS	20
#define MPORA_ML_MIN_PARTITIONS	2
#define MPORA_ML_BUDGET_COUNT	(MPORA_ML_TOTAL_PARTITIONS - MPORA_ML_MIN_PARTITIONS + 1)

/*
 * Objective selector for the secondary optimization criterion (applied after
 * minimizing tardiness as the primary criterion):
 *   0 — maximize total instructions retired (pred_next_insn)
 *   1 — minimize total predicted remaining execution time: sum of pred_rem_ns
 *       across all active tasks.
 */
#define MPORA_ML_OBJ_MIN_REM_TIME	1

#define MPORA_XGB_SCALE_NUM_INPUTS	MPORA_ML_INPUT_DIM
#define MPORA_XGB_SCALE_NUM_OUTPUTS	MPORA_ML_OUTPUT_DIM
#define MPORA_XGB_MAX_UPLOAD_SIZE	(8 * 1024 * 1024)

#define MPORA_INPUT_MAP_MAGIC	"INPMAP01"
#define MPORA_INPUT_MAP_VERSION_V1	1
#define MPORA_INPUT_MAP_VERSION_V2	2

#define MPORA_FEAT_ORDER_MAGIC	"FEATORD1"
#define MPORA_FEAT_ORDER_VERSION	1

enum mpora_xgb_output_idx {
	MPORA_XGB_OUT_REM_TIME = 0,
	MPORA_XGB_OUT_NEXT_INSN = 1,
};

struct mpora_xgb_node {
	u8 is_leaf;
	s32 fidx;
	s32 thr_q;
	s32 yes;
	s32 no;
	s32 leaf_q;
};

struct mpora_xgb_tree {
	u32 num_nodes;
	u32 root_idx;
	struct mpora_xgb_node *nodes;
};

struct mpora_xgb_ensemble {
	s32 base_q;
	s32 input_dim;
	u32 num_trees;
	struct mpora_xgb_tree *trees;
};

struct mpora_xgb_scaling {
	s32 frac_bits;
	u32 num_inputs;
	u32 num_outputs;
	s64 input_scales_q[MPORA_XGB_SCALE_NUM_INPUTS];
	s64 output_scales_q[MPORA_XGB_SCALE_NUM_OUTPUTS];
};

struct mpora_xgb_upload_buffer {
	u8 *data;
	size_t len;
	u32 received_chunks;
	u32 total_chunks;
	bool complete;
};

/*
 * Per-model feature ordering: maps model-local fidx (0..input_dim-1) to the
 * canonical feature index (0..MPORA_XGB_SCALE_NUM_INPUTS-1).  Loaded from the
 * FEATORD1 binary uploaded via MPORA_ML_UPLOAD_FEAT_ORDER0/1.
 */
struct mpora_xgb_feat_order {
	u32 input_dim;
	s32 canonical_fidx[MPORA_XGB_SCALE_NUM_INPUTS];
};

struct mpora_xgb_input_map {
	bool loaded;
	u32 num_inputs;
	u32 feature_dim;
	u32 *canonical_idx;
	u8 *valid;
	s64 *features;
};

struct mpora_xgb_model {
	bool loaded;
	bool target_loaded[2];
	bool scaling_loaded;
	bool input_map_loaded;
	bool feat_order_loaded[2];
	struct mpora_xgb_ensemble target[2];
	struct mpora_xgb_scaling scaling;
	struct mpora_xgb_input_map input_map;
	struct mpora_xgb_feat_order feat_order[2];
	struct mpora_xgb_upload_buffer upload_target[2];
	struct mpora_xgb_upload_buffer upload_scaling;
	struct mpora_xgb_upload_buffer upload_input_map;
	struct mpora_xgb_upload_buffer upload_feat_order[2];
};

struct mpora_xgb_search_ctx {
	const struct mpora_xgb_model *models[MPORA_ML_MAX_TASKS];
	const struct mpora_ml_task_features *features[MPORA_ML_MAX_TASKS];
	int num_tasks;
	int min_partitions_per_task; /* Dynamic per-step floor */
	u64 pred_rem_ns[MPORA_ML_MAX_TASKS][MPORA_ML_BUDGET_COUNT];
	u64 pred_next_insn[MPORA_ML_MAX_TASKS][MPORA_ML_BUDGET_COUNT];
	/* suffix_max_next_insn[i] = max achievable total next_insn for tasks [i..num_tasks). */
	s64 suffix_max_next_insn[MPORA_ML_MAX_TASKS + 1];
	/* suffix_min_lateness_ns[i] = min achievable total lateness for tasks [i..num_tasks). */
	s64 suffix_min_lateness_ns[MPORA_ML_MAX_TASKS + 1];
	int current_alloc[MPORA_ML_MAX_TASKS];
	int best_alloc[MPORA_ML_MAX_TASKS];
	bool found_feasible;
	bool found_any_assignment;
	u64 best_total_tardiness;
	s64 best_total_next_insn;
	s64 best_total_lateness_ns;
};

static struct mpora_xgb_model mpora_xgb_models[NUM_TASK_TYPES];
static struct mpora_xgb_search_ctx mpora_ml_search_ctx;
static bool mpora_ml_trace_enabled;
static u32 mpora_ml_trace_every_n = 10;
static u64 mpora_ml_trace_decision_count;

void mpora_ml_set_trace_enabled(bool enabled)
{
	WRITE_ONCE(mpora_ml_trace_enabled, enabled);
}

bool mpora_ml_get_trace_enabled(void)
{
	return READ_ONCE(mpora_ml_trace_enabled);
}

void mpora_ml_set_trace_every_n(u32 every_n)
{
	WRITE_ONCE(mpora_ml_trace_every_n, every_n);
}

u32 mpora_ml_get_trace_every_n(void)
{
	return READ_ONCE(mpora_ml_trace_every_n);
}

static bool mpora_ml_should_emit_trace(void)
{
	u32 every_n;
	u64 decision_idx;

	if (!mpora_ml_get_trace_enabled())
		return false;

	every_n = mpora_ml_get_trace_every_n();
	if (!every_n)
		return false;

	decision_idx = READ_ONCE(mpora_ml_trace_decision_count) + 1;
	WRITE_ONCE(mpora_ml_trace_decision_count, decision_idx);

	return (decision_idx % every_n) == 0;
}

static inline s64 mpora_sat_add_i64(s64 a, s64 b)
{
	if (b > 0 && a > S64_MAX - b)
		return S64_MAX;
	if (b < 0 && a < S64_MIN - b)
		return S64_MIN;
	return a + b;
}

static s64 mpora_div_round_closest_s64(s64 num, s64 den)
{
	s64 abs_num;
	s64 q;

	if (!den)
		return 0;

	abs_num = num < 0 ? -num : num;
	q = div64_s64(abs_num + den / 2, den);
	return num < 0 ? -q : q;
}

static void mpora_xgb_reset_upload_buffer(struct mpora_xgb_upload_buffer *buffer)
{
	if (!buffer)
		return;

	kfree(buffer->data);
	buffer->data = NULL;
	buffer->len = 0;
	buffer->received_chunks = 0;
	buffer->total_chunks = 0;
	buffer->complete = false;
}

static void mpora_xgb_reset_input_map(struct mpora_xgb_input_map *map)
{
	if (!map)
		return;

	kfree(map->canonical_idx);
	kfree(map->valid);
	kfree(map->features);
	memset(map, 0, sizeof(*map));
}

static void mpora_xgb_free_ensemble(struct mpora_xgb_ensemble *ensemble)
{
	u32 tree_idx;

	if (!ensemble)
		return;

	for (tree_idx = 0; tree_idx < ensemble->num_trees; tree_idx++)
		kfree(ensemble->trees[tree_idx].nodes);

	kfree(ensemble->trees);
	memset(ensemble, 0, sizeof(*ensemble));
}

static void mpora_xgb_reset_model(struct mpora_xgb_model *model)
{
	if (!model)
		return;

	mpora_xgb_free_ensemble(&model->target[0]);
	mpora_xgb_free_ensemble(&model->target[1]);
	mpora_xgb_reset_upload_buffer(&model->upload_target[0]);
	mpora_xgb_reset_upload_buffer(&model->upload_target[1]);
	mpora_xgb_reset_upload_buffer(&model->upload_scaling);
	mpora_xgb_reset_upload_buffer(&model->upload_input_map);
	mpora_xgb_reset_upload_buffer(&model->upload_feat_order[0]);
	mpora_xgb_reset_upload_buffer(&model->upload_feat_order[1]);
	memset(&model->scaling, 0, sizeof(model->scaling));
	mpora_xgb_reset_input_map(&model->input_map);
	memset(&model->feat_order[0], 0, sizeof(model->feat_order[0]));
	memset(&model->feat_order[1], 0, sizeof(model->feat_order[1]));
	model->target_loaded[0] = false;
	model->target_loaded[1] = false;
	model->scaling_loaded = false;
	model->input_map_loaded = false;
	model->feat_order_loaded[0] = false;
	model->feat_order_loaded[1] = false;
	model->loaded = false;
}

static int mpora_xgb_reader_u8(const u8 *buffer, size_t len, size_t *offset, u8 *out)
{
	if (!buffer || !offset || !out || *offset + sizeof(*out) > len)
		return -EINVAL;

	*out = buffer[*offset];
	*offset += sizeof(*out);
	return 0;
}

static int mpora_xgb_reader_u16(const u8 *buffer, size_t len, size_t *offset, u16 *out)
{
	if (!buffer || !offset || !out || *offset + sizeof(*out) > len)
		return -EINVAL;

	memcpy(out, buffer + *offset, sizeof(*out));
	*offset += sizeof(*out);
	return 0;
}

static int mpora_xgb_reader_u32(const u8 *buffer, size_t len, size_t *offset, u32 *out)
{
	if (!buffer || !offset || !out || *offset + sizeof(*out) > len)
		return -EINVAL;

	memcpy(out, buffer + *offset, sizeof(*out));
	*offset += sizeof(*out);
	return 0;
}

static int mpora_xgb_reader_i16(const u8 *buffer, size_t len, size_t *offset, s16 *out)
{
	if (!buffer || !offset || !out || *offset + sizeof(*out) > len)
		return -EINVAL;

	memcpy(out, buffer + *offset, sizeof(*out));
	*offset += sizeof(*out);
	return 0;
}

static int mpora_xgb_reader_i32(const u8 *buffer, size_t len, size_t *offset, s32 *out)
{
	if (!buffer || !offset || !out || *offset + sizeof(*out) > len)
		return -EINVAL;

	memcpy(out, buffer + *offset, sizeof(*out));
	*offset += sizeof(*out);
	return 0;
}

static int mpora_xgb_reader_i64(const u8 *buffer, size_t len, size_t *offset, s64 *out)
{
	if (!buffer || !offset || !out || *offset + sizeof(*out) > len)
		return -EINVAL;

	memcpy(out, buffer + *offset, sizeof(*out));
	*offset += sizeof(*out);
	return 0;
}

static int mpora_xgb_parse_ensemble(struct mpora_xgb_ensemble *ensemble,
				  const u8 *buffer,
				  size_t len)
{
	size_t offset = 0;
	s32 input_dim = 0;
	s32 base_q = 0;
	u32 num_trees = 0;
	u32 tree_idx;
	int ret;

	if (!ensemble || !buffer || !len)
		return -EINVAL;

	ret = mpora_xgb_reader_i32(buffer, len, &offset, &input_dim);
	if (ret)
		return ret;

	ret = mpora_xgb_reader_i32(buffer, len, &offset, &base_q);
	if (ret)
		return ret;

	ret = mpora_xgb_reader_u32(buffer, len, &offset, &num_trees);
	if (ret)
		return ret;

	if (input_dim <= 0 || input_dim > MPORA_XGB_SCALE_NUM_INPUTS || !num_trees)
		return -EINVAL;

	ensemble->trees = kcalloc(num_trees, sizeof(*ensemble->trees), GFP_KERNEL);
	if (!ensemble->trees)
		return -ENOMEM;

	ensemble->input_dim = input_dim;
	ensemble->base_q = base_q;
	ensemble->num_trees = num_trees;

	for (tree_idx = 0; tree_idx < num_trees; tree_idx++) {
		struct mpora_xgb_tree *tree = &ensemble->trees[tree_idx];
		u32 node_idx;
		u32 num_nodes;
		u32 root_idx;
		bool use_u16_child;
		bool use_i16_fidx;

		ret = mpora_xgb_reader_u32(buffer, len, &offset, &num_nodes);
		if (ret)
			goto parse_error;

		ret = mpora_xgb_reader_u32(buffer, len, &offset, &root_idx);
		if (ret)
			goto parse_error;

		if (!num_nodes || root_idx >= num_nodes) {
			ret = -EINVAL;
			goto parse_error;
		}

		tree->nodes = kcalloc(num_nodes, sizeof(*tree->nodes), GFP_KERNEL);
		if (!tree->nodes) {
			ret = -ENOMEM;
			goto parse_error;
		}

		tree->num_nodes = num_nodes;
		tree->root_idx = root_idx;
		use_u16_child = num_nodes <= U16_MAX;
		use_i16_fidx = input_dim <= S16_MAX;

		for (node_idx = 0; node_idx < num_nodes; node_idx++) {
			struct mpora_xgb_node *node = &tree->nodes[node_idx];
			u8 is_leaf;

			ret = mpora_xgb_reader_u8(buffer, len, &offset, &is_leaf);
			if (ret)
				goto parse_error;

			node->is_leaf = is_leaf;

			if (is_leaf) {
				ret = mpora_xgb_reader_i32(buffer, len, &offset, &node->leaf_q);
				if (ret)
					goto parse_error;
				node->fidx = -1;
				node->thr_q = 0;
				node->yes = 0;
				node->no = 0;
				continue;
			}

			if (use_i16_fidx) {
				s16 fidx16;

				ret = mpora_xgb_reader_i16(buffer, len, &offset, &fidx16);
				if (ret)
					goto parse_error;
				node->fidx = (s32)fidx16;
			} else {
				ret = mpora_xgb_reader_i32(buffer, len, &offset, &node->fidx);
				if (ret)
					goto parse_error;
			}

			ret = mpora_xgb_reader_i32(buffer, len, &offset, &node->thr_q);
			if (ret)
				goto parse_error;

			if (use_u16_child) {
				u16 child;

				ret = mpora_xgb_reader_u16(buffer, len, &offset, &child);
				if (ret)
					goto parse_error;
				node->yes = (s32)child;

				ret = mpora_xgb_reader_u16(buffer, len, &offset, &child);
				if (ret)
					goto parse_error;
				node->no = (s32)child;
			} else {
				u32 child;

				ret = mpora_xgb_reader_u32(buffer, len, &offset, &child);
				if (ret)
					goto parse_error;
				node->yes = (s32)child;

				ret = mpora_xgb_reader_u32(buffer, len, &offset, &child);
				if (ret)
					goto parse_error;
				node->no = (s32)child;
			}

			if (node->fidx < 0 || node->fidx >= input_dim ||
			    node->yes < 0 || node->yes >= num_nodes ||
			    node->no < 0 || node->no >= num_nodes) {
				ret = -EINVAL;
				goto parse_error;
			}
		}
	}

	if (offset != len)
		return -EINVAL;

	return 0;

parse_error:
	mpora_xgb_free_ensemble(ensemble);
	return ret;
}

static int mpora_xgb_parse_scaling(struct mpora_xgb_scaling *scaling,
				 const u8 *buffer,
				 size_t len)
{
	size_t offset = 0;
	u32 version;
	u32 frac_bits;
	u32 num_inputs;
	u32 num_outputs;
	u32 idx;
	int ret;

	if (!scaling || !buffer || len < 8)
		return -EINVAL;

	if (memcmp(buffer, "XGBSCL01", 8) != 0)
		return -EINVAL;

	offset = 8;

	ret = mpora_xgb_reader_u32(buffer, len, &offset, &version);
	if (ret)
		return ret;

	ret = mpora_xgb_reader_u32(buffer, len, &offset, &frac_bits);
	if (ret)
		return ret;

	ret = mpora_xgb_reader_u32(buffer, len, &offset, &num_inputs);
	if (ret)
		return ret;

	ret = mpora_xgb_reader_u32(buffer, len, &offset, &num_outputs);
	if (ret)
		return ret;

	if (version != 1 ||
	    num_inputs != MPORA_XGB_SCALE_NUM_INPUTS ||
	    num_outputs != MPORA_XGB_SCALE_NUM_OUTPUTS)
		return -EINVAL;

	scaling->frac_bits = (s32)frac_bits;
	scaling->num_inputs = num_inputs;
	scaling->num_outputs = num_outputs;

	for (idx = 0; idx < num_inputs; idx++) {
		ret = mpora_xgb_reader_i64(buffer, len, &offset,
					 &scaling->input_scales_q[idx]);
		if (ret)
			return ret;
	}

	for (idx = 0; idx < num_outputs; idx++) {
		ret = mpora_xgb_reader_i64(buffer, len, &offset,
					 &scaling->output_scales_q[idx]);
		if (ret)
			return ret;
	}

	if (offset != len)
		return -EINVAL;

	return 0;
}

static int mpora_xgb_parse_input_map(struct mpora_xgb_input_map *map,
				   const u8 *buffer,
				   size_t len)
{
	size_t offset = 0;
	u32 version;
	u32 num_inputs;
	u32 feature_dim;
	u32 idx;
	size_t features_count;
	int ret;

	if (!map || !buffer || len < 8)
		return -EINVAL;

	if (memcmp(buffer, MPORA_INPUT_MAP_MAGIC, 8) != 0)
		return -EINVAL;

	offset = 8;
	ret = mpora_xgb_reader_u32(buffer, len, &offset, &version);
	if (ret)
		return ret;
	ret = mpora_xgb_reader_u32(buffer, len, &offset, &num_inputs);
	if (ret)
		return ret;
	ret = mpora_xgb_reader_u32(buffer, len, &offset, &feature_dim);
	if (ret)
		return ret;

	if (!num_inputs || !feature_dim || feature_dim > MPORA_XGB_SCALE_NUM_INPUTS)
		return -EINVAL;

	mpora_xgb_reset_input_map(map);
	map->num_inputs = num_inputs;
	map->feature_dim = feature_dim;

	map->canonical_idx = kcalloc(feature_dim, sizeof(*map->canonical_idx), GFP_KERNEL);
	map->valid = kcalloc(num_inputs, sizeof(*map->valid), GFP_KERNEL);
	features_count = (size_t)num_inputs * feature_dim;
	map->features = kcalloc(features_count, sizeof(*map->features), GFP_KERNEL);
	if (!map->canonical_idx || !map->valid || !map->features)
		goto map_nomem;

	if (version == MPORA_INPUT_MAP_VERSION_V2) {
		for (idx = 0; idx < feature_dim; idx++) {
			ret = mpora_xgb_reader_u32(buffer, len, &offset, &map->canonical_idx[idx]);
			if (ret)
				goto map_error;
			if (map->canonical_idx[idx] >= MPORA_XGB_SCALE_NUM_INPUTS)
				goto map_error;
		}
	} else if (version == MPORA_INPUT_MAP_VERSION_V1) {
		/* Backward-compatible fallback: map values to inp_feat1..inp_featN slots. */
		for (idx = 0; idx < feature_dim; idx++)
			map->canonical_idx[idx] = 4 + idx;
	} else {
		goto map_error;
	}

	for (idx = 0; idx < num_inputs; idx++) {
		u8 valid;

		ret = mpora_xgb_reader_u8(buffer, len, &offset, &valid);
		if (ret)
			goto map_error;
		map->valid[idx] = valid ? 1 : 0;
	}

	for (idx = 0; idx < features_count; idx++) {
		ret = mpora_xgb_reader_i64(buffer, len, &offset, &map->features[idx]);
		if (ret)
			goto map_error;
	}

	if (offset != len)
		goto map_error;

	map->loaded = true;
	return 0;

map_nomem:
	ret = -ENOMEM;
map_error:
	mpora_xgb_reset_input_map(map);
	return ret;
}

static int mpora_xgb_parse_feat_order(struct mpora_xgb_feat_order *fo,
				    const u8 *buffer,
				    size_t len)
{
	size_t offset = 0;
	u32 version;
	u32 input_dim;
	u32 idx;
	int ret;

	if (!fo || !buffer || len < 8)
		return -EINVAL;

	if (memcmp(buffer, MPORA_FEAT_ORDER_MAGIC, 8) != 0)
		return -EINVAL;

	offset = 8;

	ret = mpora_xgb_reader_u32(buffer, len, &offset, &version);
	if (ret)
		return ret;

	if (version != MPORA_FEAT_ORDER_VERSION)
		return -EINVAL;

	ret = mpora_xgb_reader_u32(buffer, len, &offset, &input_dim);
	if (ret)
		return ret;

	if (!input_dim || input_dim > MPORA_XGB_SCALE_NUM_INPUTS)
		return -EINVAL;

	fo->input_dim = input_dim;
	for (idx = 0; idx < input_dim; idx++) {
		s32 canon_idx;

		ret = mpora_xgb_reader_i32(buffer, len, &offset, &canon_idx);
		if (ret)
			return ret;

		if (canon_idx < 0 || canon_idx >= (s32)MPORA_XGB_SCALE_NUM_INPUTS)
			return -EINVAL;

		fo->canonical_fidx[idx] = canon_idx;
	}

	if (offset != len)
		return -EINVAL;

	return 0;
}

static int mpora_xgb_append_upload_chunk(struct mpora_xgb_upload_buffer *upload,
					       const struct mpora_ml_model_desc *desc)
{
	u8 *new_data;
	size_t new_len;

	if (!upload || !desc)
		return -EINVAL;

	if (!desc->total_chunks ||
	    desc->payload_bytes > MPORA_ML_UPLOAD_PAYLOAD_MAX)
		return -EINVAL;

	if (desc->chunk_index == 0) {
		/* New upload stream starts at chunk 0, so discard prior partial data. */
		mpora_xgb_reset_upload_buffer(upload);
		upload->total_chunks = desc->total_chunks;
	} else if (desc->total_chunks != upload->total_chunks) {
		return -EINVAL;
	}

	if (desc->chunk_index != upload->received_chunks)
		return -EINVAL;

	new_len = upload->len + desc->payload_bytes;
	if (new_len > MPORA_XGB_MAX_UPLOAD_SIZE)
		return -E2BIG;

	new_data = krealloc(upload->data, new_len, GFP_KERNEL);
	if (!new_data)
		return -ENOMEM;

	upload->data = new_data;
	memcpy(upload->data + upload->len, desc->payload, desc->payload_bytes);
	upload->len = new_len;
	upload->received_chunks++;
	upload->complete = upload->received_chunks == upload->total_chunks;

	return 0;
}

static int mpora_xgb_predict_tree(const struct mpora_xgb_tree *tree,
				const s32 *x_q,
				s32 *out_leaf_q)
{
	s32 node_idx;
	int iter_guard = 0;

	if (!tree || !x_q || !out_leaf_q)
		return -EINVAL;

	node_idx = (s32)tree->root_idx;

	while (iter_guard++ < 100000) {
		const struct mpora_xgb_node *node;

		if (node_idx < 0 || node_idx >= tree->num_nodes)
			return -EINVAL;

		node = &tree->nodes[node_idx];
		if (node->is_leaf) {
			*out_leaf_q = node->leaf_q;
			return 0;
		}

		node_idx = (x_q[node->fidx] < node->thr_q) ? node->yes : node->no;
	}

	return -EINVAL;
}

static int mpora_xgb_predict_sample(const struct mpora_xgb_ensemble *ensemble,
				  const s32 *x_q,
				  s32 *out_q)
{
	s64 acc;
	u32 tree_idx;

	if (!ensemble || !x_q || !out_q)
		return -EINVAL;

	acc = ensemble->base_q;
	for (tree_idx = 0; tree_idx < ensemble->num_trees; tree_idx++) {
		s32 leaf_q;
		int ret;

		ret = mpora_xgb_predict_tree(&ensemble->trees[tree_idx], x_q, &leaf_q);
		if (ret)
			return ret;

		acc = mpora_sat_add_i64(acc, leaf_q);
	}

	if (acc > S32_MAX)
		acc = S32_MAX;
	if (acc < S32_MIN)
		acc = S32_MIN;

	*out_q = (s32)acc;
	return 0;
}

static s64 mpora_xgb_normalize_raw_to_q16(s64 raw_val, s64 scale_q, s32 scale_frac_bits)
{
	s64 num;
	u32 shift;

	if (scale_q <= 0)
		return 0;
	if (scale_frac_bits < 0 || scale_frac_bits > 60)
		return 0;

	shift = (u32)(16 + scale_frac_bits);
	if (shift > 60)
		return 0;

	if (raw_val > (S64_MAX >> shift))
		num = S64_MAX;
	else if (raw_val < (S64_MIN >> shift))
		num = S64_MIN;
	else
		num = raw_val << shift;

	return mpora_div_round_closest_s64(num, scale_q);
}

static s64 mpora_xgb_rescale_pred_q16_to_raw(s32 pred_q16, s64 scale_q, s32 scale_frac_bits)
{
	u64 abs_pred;
	u64 abs_scale;
	u64 den;
	u64 abs_out;
	bool neg;
	u32 shift;

	if (scale_q <= 0)
		return 0;
	if (scale_frac_bits < 0 || scale_frac_bits > 60)
		return 0;

	shift = (u32)(16 + scale_frac_bits);
	if (shift > 60)
		return 0;

	den = 1ULL << shift;
	abs_pred = pred_q16 < 0 ? (u64)(-(s64)pred_q16) : (u64)pred_q16;
	abs_scale = scale_q < 0 ? (u64)(-scale_q) : (u64)scale_q;
	neg = (pred_q16 < 0) ^ (scale_q < 0);

	abs_out = mul_u64_u64_div_u64(abs_pred, abs_scale, den);

	/*
	 * Round to nearest (matching userspace div_round_i128).
	 * mul_u64_u64_div_u64 truncates; compute the remainder via the lower
	 * 64 bits of the product (valid since shift <= 60 < 64) and round up
	 * if remainder >= den/2.
	 */
	{
		u64 rem = ((u64)abs_pred * (u64)abs_scale) & (den - 1ULL);

		if (rem >= den >> 1)
			abs_out++;
	}

	if (abs_out > (u64)S64_MAX)
		abs_out = (u64)S64_MAX;

	return neg ? -(s64)abs_out : (s64)abs_out;
}

static u64 mpora_xgb_rescale_rem_q16_to_ns(s32 pred_q16, s64 scale_q, s32 scale_frac_bits)
{
	u64 den;
	u32 shift;

	if (scale_q <= 0)
		return 0;
	if (scale_frac_bits < 0 || scale_frac_bits > 60)
		return 0;

	shift = (u32)(16 + scale_frac_bits);
	if (shift > 60)
		return 0;

	if (pred_q16 <= 0 || scale_q <= 0)
		return 0;

	den = 1ULL << shift;

	/*
	 * Compute (pred_q16 * scale_q * 1e9) / den directly in nanoseconds,
	 * matching the userspace rescale_rem_time_q16_to_ns().  Multiply
	 * pred_q16 by 1e9 first: pred_q16 <= S32_MAX < 2^31, so
	 * pred_q16 * 1e9 < 2^62, which fits safely in u64.
	 * mul_u64_u64_div_u64 carries a 128-bit intermediate product.
	 */
	return mul_u64_u64_div_u64((u64)pred_q16 * 1000000000ULL,
				   (u64)scale_q, den);
}

static void mpora_xgb_fill_canonical_features(const struct mpora_ml_task_features *feat,
					   int budget,
					   s64 out_features[MPORA_XGB_SCALE_NUM_INPUTS])
{
	int idx;

	for (idx = 0; idx < MPORA_XGB_SCALE_NUM_INPUTS; idx++)
		out_features[idx] = 0;

	if (!feat)
		return;

	/* Canonical feature layout mirrors userspace XGB implementation README. */
	out_features[0] = (s64)feat->insn_sum;
	out_features[1] = (s64)feat->prev_cache_partitions_lag[0];
	out_features[2] = (s64)budget;
	out_features[3] = (s64)feat->prev_insn_delta_lag[0];

	/* inp_feat1..inp_feat6 (canonical 4..9) are filled by
	 * mpora_xgb_apply_input_map_features() after this function returns;
	 * they are left at zero here so the caller can safely overwrite them. */

	/*
	 * Training convention (preprocess.py::calculate_prev_data):
	 *   prev_insn_N = shift(N) of prev_insn = the instruction delta from N
	 *   periods ago = lag[N].  prev_insn (canonical 3) is lag[0], so
	 *   prev_insn_1 (canonical 10) must be lag[1], not lag[0].
	 *
	 *   With MPORA_HISTORY_LAG == 6 we store lag[0..5]; lag[1..5] map to
	 *   prev_insn_1..prev_insn_5 (canonical 10..14) and similarly for
	 *   prev_budget_1..prev_budget_5 (canonical 15..19).
	 */
	for (idx = 0; idx < MPORA_HISTORY_LAG - 1; idx++) {
		out_features[10 + idx] = (s64)feat->prev_insn_delta_lag[idx + 1];
		out_features[15 + idx] = (s64)feat->prev_cache_partitions_lag[idx + 1];
	}
}

static void mpora_xgb_apply_input_map_features(const struct mpora_xgb_model *model,
					     const struct mpora_ml_task_features *feat,
					     s64 out_features[MPORA_XGB_SCALE_NUM_INPUTS])
{
	const struct mpora_xgb_input_map *map;
	u32 input_num;
	size_t row_base;
	u32 idx;

	if (!model || !feat || !out_features)
		return;

	map = &model->input_map;
	if (!map->loaded || !map->num_inputs || !map->feature_dim)
		return;

	input_num = feat->current_input_num;
	if (input_num >= map->num_inputs)
		return;
	if (!map->valid[input_num])
		return;

	row_base = (size_t)input_num * map->feature_dim;
	for (idx = 0; idx < map->feature_dim; idx++) {
		u32 canon_idx = map->canonical_idx[idx];

		if (canon_idx >= MPORA_XGB_SCALE_NUM_INPUTS)
			continue;
		out_features[canon_idx] = map->features[row_base + idx];
	}
}

static void mpora_xgb_fill_trace_features(const struct mpora_xgb_model *model,
					const struct mpora_ml_task_features *feat,
					s64 out_features[MPORA_XGB_SCALE_NUM_INPUTS])
{
	if (!out_features)
		return;

	mpora_xgb_fill_canonical_features(feat, 0, out_features);
	if (model)
		mpora_xgb_apply_input_map_features(model, feat, out_features);

	/* Userspace allocator overwrites next_budget per candidate budget anyway. */
	out_features[2] = 0;
}

static void mpora_xgb_trace_request(const struct mpora_xgb_search_ctx *ctx, bool trace_now)
{
	int task_idx;

	if (!ctx || !trace_now)
		return;

	pr_info("MPORA_ML_TRACE argv_prefix --argv %d\n", ctx->num_tasks);
	for (task_idx = 0; task_idx < ctx->num_tasks; task_idx++) {
		s64 feature_raw[MPORA_XGB_SCALE_NUM_INPUTS];
		const struct mpora_xgb_model *model = ctx->models[task_idx];
		const struct mpora_ml_task_features *feat = ctx->features[task_idx];

		mpora_xgb_fill_trace_features(model, feat, feature_raw);
		pr_info("MPORA_ML_TRACE argv_job[%d] %u %llu %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
			task_idx,
			feat ? feat->workload_id : 0,
			feat ? (unsigned long long)feat->deadline_remaining_ns : 0ULL,
			(long long)feature_raw[0],
			(long long)feature_raw[1],
			(long long)feature_raw[2],
			(long long)feature_raw[3],
			(long long)feature_raw[4],
			(long long)feature_raw[5],
			(long long)feature_raw[6],
			(long long)feature_raw[7],
			(long long)feature_raw[8],
			(long long)feature_raw[9],
			(long long)feature_raw[10],
			(long long)feature_raw[11],
			(long long)feature_raw[12],
			(long long)feature_raw[13],
			(long long)feature_raw[14],
			(long long)feature_raw[15],
			(long long)feature_raw[16],
			(long long)feature_raw[17],
			(long long)feature_raw[18],
			(long long)feature_raw[19]);
	}
}

static void mpora_xgb_trace_result(const struct mpora_xgb_search_ctx *ctx,
					 int *allocations,
					 int ret,
					 u64 runtime_ns,
					 bool trace_now)
{
	int task_idx;

	if (!ctx || !allocations || !trace_now)
		return;

	pr_info("MPORA_ML_TRACE alloc ret=%d feasible=%d total_next_insn=%lld total_tardiness_ns=%llu\n",
		ret,
		ctx->best_total_tardiness == 0 ? 1 : 0,
		(long long)ctx->best_total_next_insn,
		(unsigned long long)ctx->best_total_tardiness);
	pr_info("MPORA_ML_TRACE alloc_runtime_ns=%llu\n",
		(unsigned long long)runtime_ns);

	for (task_idx = 0; task_idx < ctx->num_tasks; task_idx++)
		pr_info("MPORA_ML_TRACE alloc_job[%d] %d\n", task_idx, allocations[task_idx]);
}

/*
 * Build a model-local-indexed input vector for one target ensemble.
 *
 * The ensemble's tree nodes use fidx values in [0, ensemble->input_dim).
 * feat_order->canonical_fidx[mf] gives the canonical feature index for
 * model-local index mf, matching the userspace prepare_budget_predictions()
 * in xgb_utils_userspace.c.
 */
static void mpora_xgb_build_local_x_q(const struct mpora_xgb_feat_order *feat_order,
				     const struct mpora_xgb_scaling *scaling,
				     const s64 *feature_raw,
				     s32 *x_q_out)
{
	u32 mf;

	for (mf = 0; mf < feat_order->input_dim; mf++) {
		s32 canon_idx = feat_order->canonical_fidx[mf];
		s64 scale_q = scaling->input_scales_q[canon_idx];
		s64 norm_q16;

		norm_q16 = mpora_xgb_normalize_raw_to_q16(feature_raw[canon_idx],
							 scale_q,
							 scaling->frac_bits);
		if (norm_q16 > S32_MAX)
			norm_q16 = S32_MAX;
		if (norm_q16 < S32_MIN)
			norm_q16 = S32_MIN;
		x_q_out[mf] = (s32)norm_q16;
	}
}

static int mpora_xgb_prepare_predictions(struct mpora_xgb_search_ctx *ctx)
{
	int task_idx;

	if (!ctx)
		return -EINVAL;

	for (task_idx = 0; task_idx < ctx->num_tasks; task_idx++) {
		const struct mpora_xgb_model *model = ctx->models[task_idx];
		const struct mpora_ml_task_features *feat = ctx->features[task_idx];
		s64 base_feature_raw[MPORA_XGB_SCALE_NUM_INPUTS];
		s32 x_q_rem[MPORA_XGB_SCALE_NUM_INPUTS];
		s32 x_q_insn[MPORA_XGB_SCALE_NUM_INPUTS];
		int rem_budget_fidx = -1, insn_budget_fidx = -1;
		bool map_sets_budget = false;
		int budget;
		u32 mf;

		if (!model || !feat || !model->loaded)
			return -EINVAL;

		/*
		 * Build the budget-invariant feature vector once per task.
		 * mpora_xgb_fill_canonical_features places budget at canonical
		 * index 2; mpora_xgb_apply_input_map_features may override any
		 * canonical index, including 2, with per-input static values.
		 */
		mpora_xgb_fill_canonical_features(feat, 0, base_feature_raw);
		mpora_xgb_apply_input_map_features(model, feat, base_feature_raw);

		/*
		 * Detect whether the input map provides canonical index 2
		 * (the budget feature).  If so, the budget value has no effect
		 * on the predictions and we skip the per-budget x_q patch.
		 */
		{
			const struct mpora_xgb_input_map *map = &model->input_map;
			u32 inp = feat->current_input_num;

			if (map->loaded && map->num_inputs &&
			    inp < map->num_inputs && map->valid[inp]) {
				for (mf = 0; mf < map->feature_dim; mf++) {
					if (map->canonical_idx[mf] == 2) {
						map_sets_budget = true;
						break;
					}
				}
			}
		}

		/*
		 * Build base x_q vectors from the budget-invariant features.
		 * The rem_time and next_insn targets may use different feature
		 * subsets and orderings.
		 */
		mpora_xgb_build_local_x_q(&model->feat_order[MPORA_XGB_OUT_REM_TIME],
					 &model->scaling, base_feature_raw, x_q_rem);
		mpora_xgb_build_local_x_q(&model->feat_order[MPORA_XGB_OUT_NEXT_INSN],
					 &model->scaling, base_feature_raw, x_q_insn);

		/*
		 * Find which model-local index corresponds to canonical index 2
		 * in each target's feature ordering.  Only needed when the
		 * input map does not already fix the budget feature.
		 */
		if (!map_sets_budget) {
			const struct mpora_xgb_feat_order *fo_rem =
				&model->feat_order[MPORA_XGB_OUT_REM_TIME];
			const struct mpora_xgb_feat_order *fo_insn =
				&model->feat_order[MPORA_XGB_OUT_NEXT_INSN];

			for (mf = 0; mf < fo_rem->input_dim; mf++) {
				if (fo_rem->canonical_fidx[mf] == 2) {
					rem_budget_fidx = (int)mf;
					break;
				}
			}
			for (mf = 0; mf < fo_insn->input_dim; mf++) {
				if (fo_insn->canonical_fidx[mf] == 2) {
					insn_budget_fidx = (int)mf;
					break;
				}
			}
		}

		for (budget = MPORA_ML_MIN_PARTITIONS;
		     budget <= MPORA_ML_TOTAL_PARTITIONS;
		     budget++) {
			s32 rem_q16;
			s32 next_insn_q16;
			s64 next_insn_raw;
			u64 rem_ns;
			int budget_idx = budget - MPORA_ML_MIN_PARTITIONS;
			int ret;

			/*
			 * Patch only the budget-varying entry in x_q rather
			 * than rebuilding the whole vector from scratch.
			 */
			if (rem_budget_fidx >= 0 || insn_budget_fidx >= 0) {
				s64 norm_q16 = mpora_xgb_normalize_raw_to_q16(
					(s64)budget,
					model->scaling.input_scales_q[2],
					model->scaling.frac_bits);

				if (norm_q16 > S32_MAX)
					norm_q16 = S32_MAX;
				if (norm_q16 < S32_MIN)
					norm_q16 = S32_MIN;
				if (rem_budget_fidx >= 0)
					x_q_rem[rem_budget_fidx] = (s32)norm_q16;
				if (insn_budget_fidx >= 0)
					x_q_insn[insn_budget_fidx] = (s32)norm_q16;
			}

			ret = mpora_xgb_predict_sample(&model->target[MPORA_XGB_OUT_REM_TIME],
						     x_q_rem,
						     &rem_q16);
			if (ret)
				return ret;

			ret = mpora_xgb_predict_sample(&model->target[MPORA_XGB_OUT_NEXT_INSN],
						     x_q_insn,
						     &next_insn_q16);
			if (ret)
				return ret;

			rem_ns = mpora_xgb_rescale_rem_q16_to_ns(
				rem_q16,
				model->scaling.output_scales_q[MPORA_XGB_OUT_REM_TIME],
				model->scaling.frac_bits);
			next_insn_raw = mpora_xgb_rescale_pred_q16_to_raw(
				next_insn_q16,
				model->scaling.output_scales_q[MPORA_XGB_OUT_NEXT_INSN],
				model->scaling.frac_bits);
			if (next_insn_raw < 0)
				next_insn_raw = 0;

			ctx->pred_rem_ns[task_idx][budget_idx] = rem_ns;
			ctx->pred_next_insn[task_idx][budget_idx] = (u64)next_insn_raw;
		}
	}

	return 0;
}

static void mpora_xgb_evaluate_assignment(struct mpora_xgb_search_ctx *ctx,
					 s64 total_next_insn,
					 s64 total_lateness_ns)
{
	u64 total_tardiness = 0;
	bool feasible = true;
	int task_idx;

	for (task_idx = 0; task_idx < ctx->num_tasks; task_idx++) {
		int budget_idx = ctx->current_alloc[task_idx] - MPORA_ML_MIN_PARTITIONS;
		u64 pred_ns = ctx->pred_rem_ns[task_idx][budget_idx];
		u64 deadline_ns = ctx->features[task_idx]->deadline_remaining_ns;

		if (!deadline_ns)
			deadline_ns = U64_MAX;

		if (pred_ns > deadline_ns) {
			feasible = false;
			total_tardiness += pred_ns - deadline_ns;
		}
	}

	ctx->found_any_assignment = true;

	if (ctx->found_feasible) {
		if (!feasible)
			return;

#if MPORA_ML_OBJ_MIN_REM_TIME
		if (total_lateness_ns < ctx->best_total_lateness_ns) {
#else
		if (total_next_insn > ctx->best_total_next_insn) {
#endif
			memcpy(ctx->best_alloc,
			       ctx->current_alloc,
			       sizeof(int) * ctx->num_tasks);
			ctx->best_total_next_insn = total_next_insn;
			ctx->best_total_lateness_ns = total_lateness_ns;
			ctx->best_total_tardiness = 0;
		}
		return;
	}

	if (feasible) {
		ctx->found_feasible = true;
		memcpy(ctx->best_alloc,
		       ctx->current_alloc,
		       sizeof(int) * ctx->num_tasks);
		ctx->best_total_next_insn = total_next_insn;
		ctx->best_total_lateness_ns = total_lateness_ns;
		ctx->best_total_tardiness = 0;
		return;
	}

#if MPORA_ML_OBJ_MIN_REM_TIME
	if (total_tardiness < ctx->best_total_tardiness ||
	    (total_tardiness == ctx->best_total_tardiness &&
	     total_lateness_ns < ctx->best_total_lateness_ns)) {
#else
	if (total_tardiness < ctx->best_total_tardiness ||
	    (total_tardiness == ctx->best_total_tardiness &&
	     total_next_insn > ctx->best_total_next_insn)) {
#endif
		memcpy(ctx->best_alloc,
		       ctx->current_alloc,
		       sizeof(int) * ctx->num_tasks);
		ctx->best_total_next_insn = total_next_insn;
		ctx->best_total_lateness_ns = total_lateness_ns;
		ctx->best_total_tardiness = total_tardiness;
	}
}

static void mpora_xgb_dfs_assign(struct mpora_xgb_search_ctx *ctx,
	   int task_idx,
	   int used_budget,
	   s64 partial_next_insn,
	   s64 partial_lateness_ns)
{
	int remaining;
	int min_remaining;
	int max_budget_for_task;
	int min_budget_for_task;
	int budget;

       if (task_idx == ctx->num_tasks) {
	       if (used_budget != MPORA_ML_TOTAL_PARTITIONS)
		       return;
	       mpora_xgb_evaluate_assignment(ctx, partial_next_insn,
					   partial_lateness_ns);
	       return;
       }

       /*
	* Feasible-phase bound pruning: prune subtrees that cannot improve on
	* the current best secondary objective.
	*
	* MPORA_ML_OBJ_MIN_REM_TIME=0: upper-bound on next_insn — if even the
	* maximum possible next_insn for remaining tasks can't beat the best,
	* skip this subtree.
	*
	* MPORA_ML_OBJ_MIN_REM_TIME=1: lower-bound on lateness — if even the
	* minimum possible lateness for remaining tasks can't beat the best,
	* skip this subtree.
	*/
       if (ctx->found_feasible) {
#if MPORA_ML_OBJ_MIN_REM_TIME
	       s64 lower_bound = mpora_sat_add_i64(partial_lateness_ns,
						 ctx->suffix_min_lateness_ns[task_idx]);

	       if (lower_bound >= ctx->best_total_lateness_ns)
		       return;
#else
	       s64 upper_bound = mpora_sat_add_i64(partial_next_insn,
						 ctx->suffix_max_next_insn[task_idx]);

	       if (upper_bound <= ctx->best_total_next_insn)
		       return;
#endif
       }

       remaining = ctx->num_tasks - (task_idx + 1);
       min_remaining = remaining * ctx->min_partitions_per_task;
       max_budget_for_task = MPORA_ML_TOTAL_PARTITIONS - (used_budget + min_remaining);
       if (max_budget_for_task > MPORA_ML_TOTAL_PARTITIONS)
	       max_budget_for_task = MPORA_ML_TOTAL_PARTITIONS;
       min_budget_for_task = ctx->min_partitions_per_task;
       if (max_budget_for_task < min_budget_for_task)
	       return;

	       for (budget = min_budget_for_task;
		    budget <= max_budget_for_task;
		    budget++) {
		       int budget_idx = budget - MPORA_ML_MIN_PARTITIONS;
		       s64 new_partial_next_insn = mpora_sat_add_i64(partial_next_insn,
			       (s64)ctx->pred_next_insn[task_idx][budget_idx]);
		       s64 new_partial_lateness_ns = mpora_sat_add_i64(partial_lateness_ns,
			       (s64)ctx->pred_rem_ns[task_idx][budget_idx]);

		       ctx->current_alloc[task_idx] = budget;
		       mpora_xgb_dfs_assign(ctx, task_idx + 1, used_budget + budget,
			       new_partial_next_insn, new_partial_lateness_ns);
	       }
}

int mpora_ml_load_model(const struct mpora_ml_model_desc *desc)
{
	struct mpora_xgb_model *model;
	struct mpora_xgb_upload_buffer *upload;
	int target_idx;
	int ret;

	if (!desc || desc->workload_id >= NUM_TASK_TYPES)
		return -EINVAL;

	model = &mpora_xgb_models[desc->workload_id];

	if (desc->upload_kind == MPORA_ML_UPLOAD_XGB_TARGET0 ||
	    desc->upload_kind == MPORA_ML_UPLOAD_XGB_TARGET1) {
		target_idx = desc->upload_kind == MPORA_ML_UPLOAD_XGB_TARGET0 ? 0 : 1;
		upload = &model->upload_target[target_idx];
	} else if (desc->upload_kind == MPORA_ML_UPLOAD_XGB_SCALING) {
		target_idx = -1;
		upload = &model->upload_scaling;
	} else if (desc->upload_kind == MPORA_ML_UPLOAD_INPUT_MAP) {
		target_idx = -2;
		upload = &model->upload_input_map;
	} else if (desc->upload_kind == MPORA_ML_UPLOAD_FEAT_ORDER0 ||
		   desc->upload_kind == MPORA_ML_UPLOAD_FEAT_ORDER1) {
		target_idx = desc->upload_kind == MPORA_ML_UPLOAD_FEAT_ORDER0 ? -3 : -4;
		upload = &model->upload_feat_order[target_idx == -3 ? 0 : 1];
	} else {
		return -EINVAL;
	}

	ret = mpora_xgb_append_upload_chunk(upload, desc);
	if (ret)
		return ret;

	if (!upload->complete)
		return 0;

	if (target_idx >= 0) {
		mpora_xgb_free_ensemble(&model->target[target_idx]);
		ret = mpora_xgb_parse_ensemble(&model->target[target_idx],
					     upload->data,
					     upload->len);
		if (ret)
			return ret;
		model->target_loaded[target_idx] = true;
	} else if (target_idx == -1) {
		ret = mpora_xgb_parse_scaling(&model->scaling, upload->data, upload->len);
		if (ret)
			return ret;
		model->scaling_loaded = true;
	} else if (target_idx == -2) {
		ret = mpora_xgb_parse_input_map(&model->input_map, upload->data, upload->len);
		if (ret)
			return ret;
		model->input_map_loaded = true;
	} else {
		int fo_idx = (target_idx == -3) ? 0 : 1;

		ret = mpora_xgb_parse_feat_order(&model->feat_order[fo_idx],
					       upload->data, upload->len);
		if (ret)
			return ret;
		model->feat_order_loaded[fo_idx] = true;
	}

	/* Free upload payload once parsed successfully. */
	mpora_xgb_reset_upload_buffer(upload);

	/*
	 * Mark workload as allocatable only when all required artifacts are
	 * loaded.  Also validate that feat_order input_dim matches the
	 * corresponding ensemble input_dim to catch mismatched exports early.
	 */
	if (model->target_loaded[0] && model->target_loaded[1] &&
	    model->scaling_loaded && model->input_map_loaded &&
	    model->feat_order_loaded[0] && model->feat_order_loaded[1]) {
		if (model->feat_order[0].input_dim != (u32)model->target[0].input_dim ||
		    model->feat_order[1].input_dim != (u32)model->target[1].input_dim)
			return -EINVAL;
		model->loaded = true;
	}

	return 0;
}

int mpora_ml_try_allocate(const struct mpora_ml_request *req, int *allocations)
{
       struct mpora_xgb_search_ctx *ctx;
       bool trace_now;
       u64 alloc_start_ns;
       u64 alloc_end_ns;
       u64 alloc_runtime_ns;
       int i;
       int ret;

       if (!req || !allocations)
	       return -EINVAL;
       if (!req->num_tasks || req->num_tasks > MPORA_ML_MAX_TASKS)
	       return -EINVAL;
       if (req->num_tasks * MPORA_ML_MIN_PARTITIONS > MPORA_ML_TOTAL_PARTITIONS)
	       return -EINVAL;

       ctx = &mpora_ml_search_ctx;
       memset(ctx, 0, sizeof(*ctx));

       ctx->num_tasks = req->num_tasks;
       /* Dynamic per-step floor: max(MPORA_ML_MIN_PARTITIONS, MPORA_ML_TOTAL_PARTITIONS / (2 * num_tasks)) */
       {
	       int even_split = MPORA_ML_TOTAL_PARTITIONS / (2 * ctx->num_tasks);
	       ctx->min_partitions_per_task = (even_split > MPORA_ML_MIN_PARTITIONS) ? even_split : MPORA_ML_MIN_PARTITIONS;
       }
       ctx->best_total_tardiness = U64_MAX;
       ctx->best_total_next_insn = S64_MIN;
       ctx->best_total_lateness_ns = S64_MAX;

       for (i = 0; i < req->num_tasks; i++) {
	       u32 workload = req->task[i].workload_id;
	       const struct mpora_xgb_model *model;

	       if (workload >= NUM_TASK_TYPES)
		       return -EINVAL;

	       model = &mpora_xgb_models[workload];
	       if (!model->loaded)
		       return -ENOENT;

	       ctx->models[i] = model;
	       ctx->features[i] = &req->task[i];
       }

       trace_now = mpora_ml_should_emit_trace();
       mpora_xgb_trace_request(ctx, trace_now);
       alloc_start_ns = ktime_get_ns();

       ret = mpora_xgb_prepare_predictions(ctx);
       if (ret)
	       goto out;

       /* Build suffix tables for feasible-phase DFS pruning. */
       ctx->suffix_max_next_insn[ctx->num_tasks] = 0;
       ctx->suffix_min_lateness_ns[ctx->num_tasks] = 0;
       for (i = ctx->num_tasks - 1; i >= 0; i--) {
	       u64 task_max_insn = 0;
	       u64 task_min_pred = U64_MAX;
	       int bi;

	       for (bi = 0; bi < MPORA_ML_BUDGET_COUNT; bi++) {
		       if (ctx->pred_next_insn[i][bi] > task_max_insn)
			       task_max_insn = ctx->pred_next_insn[i][bi];
		       if (ctx->pred_rem_ns[i][bi] < task_min_pred)
			       task_min_pred = ctx->pred_rem_ns[i][bi];
	       }
	       ctx->suffix_max_next_insn[i] = mpora_sat_add_i64(
		       ctx->suffix_max_next_insn[i + 1], (s64)task_max_insn);
	       ctx->suffix_min_lateness_ns[i] = mpora_sat_add_i64(
		       ctx->suffix_min_lateness_ns[i + 1], (s64)task_min_pred);
       }

       mpora_xgb_dfs_assign(ctx, 0, 0, 0, 0);

       if (!ctx->found_any_assignment) {
	       ret = -EINVAL;
	       goto out;
       }

       for (i = 0; i < MPORA_ML_MAX_TASKS; i++)
	       allocations[i] = MPORA_ML_MIN_PARTITIONS;

       for (i = 0; i < req->num_tasks; i++)
	       allocations[i] = ctx->best_alloc[i];

       alloc_end_ns = ktime_get_ns();
       alloc_runtime_ns = alloc_end_ns - alloc_start_ns;

       ret = 0;
       mpora_xgb_trace_result(ctx, allocations, ret, alloc_runtime_ns, trace_now);
out:
       if (ret) {
	       alloc_end_ns = ktime_get_ns();
	       alloc_runtime_ns = alloc_end_ns - alloc_start_ns;
	       mpora_xgb_trace_result(ctx, allocations, ret, alloc_runtime_ns, trace_now);
       }
       return ret;
}

void mpora_ml_init(void)
{
	int i;

	/* Ensure stale models from previous module loads are dropped cleanly. */
	for (i = 0; i < NUM_TASK_TYPES; i++)
		mpora_xgb_reset_model(&mpora_xgb_models[i]);
}

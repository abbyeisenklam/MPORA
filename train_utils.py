import torch
import os
import ctypes
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import time
import json
import warnings
from sklearn.cluster import KMeans
from plotting_utils import *

warnings.filterwarnings('ignore')


def load_and_prepare_data(task, input_features, raw_data_file_path, normalized_data_file_path, random_split=False, train_split=0.1, plot_train_runs=False, debug_plots=False, delta_tag=None):

    if random_split:
        df = pd.read_csv(normalized_data_file_path)
        output_features = ['rem_time', 'next_insn']
        X = df[input_features].values.astype(np.float32)
        y = df[output_features].values.astype(np.float32)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        split = {}

        next_insn_max = df['next_insn'].max()
        rem_time_max = df['rem_time'].max()

        print(f"Dataset shape: {X.shape}, Target shape: {y.shape}")

    else:
        splits_dir = 'splits'
        splits_file = os.path.join(splits_dir, f'{task}.txt')
        output_features = ['rem_time', 'next_insn']
        df = pd.read_csv(normalized_data_file_path)

        if os.path.exists(splits_file):
            split_map = {}
            with open(splits_file, 'r') as _sf:
                for _line in _sf:
                    _line = _line.strip()
                    if not _line:
                        continue
                    _cat, _names_str = _line.split(':', 1)
                    split_map[_cat] = [n.strip() for n in _names_str.split(',') if n.strip()]
            train_inp_names = split_map.get('train', [])
            validation_inp_names = split_map.get('validation', [])
            test_inp_names = split_map.get('test', [])
            print(f"Loaded train/test/validation splits from {splits_file}")
        else:
            raw_df = pd.read_csv(raw_data_file_path)

            grouped = raw_df.groupby('run_id').agg(
                prev_budget_min=('prev_budget', 'min'),
                prev_budget_max=('prev_budget', 'max'),
                next_budget_min=('next_budget', 'min'),
                next_budget_max=('next_budget', 'max'),
            )
            valid_run_ids = grouped[
                (grouped['prev_budget_min'] == 5) & (grouped['prev_budget_max'] == 5) &
                (grouped['next_budget_min'] == 5) & (grouped['next_budget_max'] == 5)
            ].index

            reference_df = raw_df[raw_df['run_id'].isin(valid_run_ids)].copy()

            execution_time_df = (
                reference_df.groupby(['inp_name', 'run_id'])['insn_sum']
                .max()
                .reset_index()
            )
            execution_time_df.rename(columns={'insn_sum': 'execution_time'}, inplace=True)

            assert len(execution_time_df) > 0, "No data found with prev_budget == next_budget == 5"

            inp_exec = (
                execution_time_df.groupby('inp_name')['execution_time']
                .median()
                .sort_values()
                .reset_index()
            )

            num_inputs = int(execution_time_df['inp_name'].nunique())
            desired_clusters = int(train_split * num_inputs)
            n_samples = len(execution_time_df)
            n_clusters = max(1, min(desired_clusters, n_samples))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            execution_time_df['cluster'] = kmeans.fit_predict(execution_time_df[['execution_time']])

            selected_pairs = []
            used_inp_names = set()
            for cluster_id, group in sorted(execution_time_df.groupby('cluster'), key=lambda kv: kv[0]):
                counts = group['inp_name'].value_counts()
                chosen = next((candidate for candidate in counts.index if candidate not in used_inp_names), None)
                if chosen is None:
                    chosen = counts.index[0]
                used_inp_names.add(chosen)
                selected_pairs.append((cluster_id, chosen))

            train_inp_names = [str(inp) for _, inp in sorted(selected_pairs, key=lambda x: x[0])]

            unique_inp_names = execution_time_df['inp_name'].unique().tolist()
            train_set = set(train_inp_names)
            remaining_inp_names = [name for name in unique_inp_names if name not in train_set]

            val_pairs = []
            used_val_inp_names = set()
            for cluster_id, group in sorted(execution_time_df.groupby('cluster'), key=lambda kv: kv[0]):
                counts = group['inp_name'].value_counts()
                chosen = None
                for candidate in counts.index:
                    if candidate not in train_inp_names and candidate not in used_val_inp_names:
                        chosen = candidate
                        break
                if chosen is None:
                    remaining_candidates = [c for c in unique_inp_names if c not in train_inp_names and c not in used_val_inp_names]
                    if len(remaining_candidates) > 0:
                        chosen = remaining_candidates[0]
                    else:
                        print("[load_and_prepare_data] Warning: Not enough distinct inp_names to fill validation set; allowing overlap.")
                        chosen = counts.index[0]
                used_val_inp_names.add(chosen)
                val_pairs.append((cluster_id, chosen))

            validation_inp_names = [str(inp) for _, inp in sorted(val_pairs, key=lambda x: x[0])]

            all_inp_names = raw_df['inp_name'].unique()
            test_inp_names = [str(name) for name in all_inp_names
                    if str(name) not in train_inp_names and str(name) not in validation_inp_names]

            print(f"Train inp_names: {train_inp_names}")
            print(f"Validation inp_names: {validation_inp_names}")
            print(f"Test inp_names: {test_inp_names}")

            if debug_plots:
                plot_inp_feat_vs_targets(df, input_features)
                plot_comparison_inputs(task, df, ['input19', 'input56'])

            cal_fraction = 0.4
            wt_fraction = 0.4
            n_test_names = len(test_inp_names)
            n_cal = int(round(cal_fraction * n_test_names))
            n_wt = int(round(wt_fraction * n_test_names))
            n_cal = max(0, min(n_cal, n_test_names))
            n_wt = max(0, min(n_wt, max(n_test_names - n_cal, 0)))

            cal_inp_names = test_inp_names[:n_cal]
            wt_inp_names = test_inp_names[n_cal:n_cal + n_wt]
            test_cp_inp_names = test_inp_names[n_cal + n_wt:]

            os.makedirs(splits_dir, exist_ok=True)
            with open(splits_file, 'w') as _sf:
                _sf.write(f"train:{','.join(train_inp_names)}\n")
                _sf.write(f"validation:{','.join(validation_inp_names)}\n")
                _sf.write(f"test:{','.join(test_inp_names)}\n")
                _sf.write(f"calibration:{','.join(cal_inp_names)}\n")
                _sf.write(f"weights:{','.join(wt_inp_names)}\n")
                _sf.write(f"test_cp:{','.join(test_cp_inp_names)}\n")
            print(f"Saved splits to {splits_file}")

        X = df[input_features].values.astype(np.float32)
        y = df[output_features].values.astype(np.float32)
        split = {}
        split['train'] = train_inp_names
        split['validation'] = validation_inp_names
        split['test'] = test_inp_names

        if plot_train_runs:
            plot_train_run_profiles(task, df, train_inp_names)

        X_train = X[df['inp_name'].isin(train_inp_names)]
        y_train = y[df['inp_name'].isin(train_inp_names)]
        X_val = X[df['inp_name'].isin(validation_inp_names)]
        y_val = y[df['inp_name'].isin(validation_inp_names)]
        X_test = X[df['inp_name'].isin(test_inp_names)]
        y_test = y[df['inp_name'].isin(test_inp_names)]

        next_insn_max = None
        rem_time_max = None
        try:
            norm_dir = os.path.dirname(os.path.abspath(normalized_data_file_path))
            _norm_stem = os.path.splitext(os.path.basename(normalized_data_file_path))[0]
            _delta_tag = _norm_stem[len('normalized_data_'):] if _norm_stem.startswith('normalized_data_') else ''
            _pkl_name = f'output_max_values_{_delta_tag}.pkl' if _delta_tag else 'output_max_values.pkl'
            with open(os.path.join(norm_dir, _pkl_name), 'rb') as f:
                out_max = pickle.load(f)
                if isinstance(out_max, dict) and 'next_insn' in out_max:
                    next_insn_max = float(out_max['next_insn'])
                if isinstance(out_max, dict) and 'rem_time' in out_max:
                    rem_time_max = float(out_max['rem_time'])
        except Exception:
            next_insn_max = None
            rem_time_max = None

    return X_train, X_val, X_test, y_train, y_val, y_test, split, next_insn_max, rem_time_max


def train_and_evaluate_xgboost(
    xgb_model_path: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    use_gpu: bool | None = None,
    xgb_params: dict | None = None,
    verbose: bool = True,
    feature_names: list[str] | None = None,
):
    """Train and evaluate XGBoost (one booster per target). Data assumed pre-scaled to [0,1]."""
    input_dim = int(X_train.shape[1])
    target_dim = int(y_train.shape[1])
    assert X_val.shape[1] == input_dim and X_test.shape[1] == input_dim, "X_val/X_test must match X_train feature dimension"
    assert y_val.shape[1] == target_dim and y_test.shape[1] == target_dim, "y_val/y_test must match y_train target dimension"

    if os.path.isfile(xgb_model_path):
        if verbose:
            print(f"Loading existing XGBoost model from: {xgb_model_path}")
        xgb_models = []
        for i in range(target_dim):
            model_file_i = f"{xgb_model_path}_target_{i}.model"
            booster = xgb.Booster()
            booster.load_model(model_file_i)
            xgb_models.append(booster)

        dtest_full = xgb.DMatrix(X_test)
        if feature_names is not None and len(feature_names) == int(X_test.shape[1]):
            rem_mask = np.ones((int(X_test.shape[1])), dtype=bool)
        X_test_rem = X_test[:, rem_mask]
        dtest_rem = xgb.DMatrix(X_test_rem)

        start = time.perf_counter()
        preds_list = []
        for i, booster in enumerate(xgb_models):
            dtest_i = dtest_rem if i == 0 else dtest_full
            preds_list.append(booster.predict(dtest_i))
        xgb_preds = np.column_stack(preds_list).astype(np.float32)
        end = time.perf_counter()
        xgb_total_ms = (end - start) * 1000.0
        xgb_ms_per_sample = xgb_total_ms / max(X_test.shape[0], 1)

        mse_list = [mean_squared_error(y_test[:, i], xgb_preds[:, i]) for i in range(target_dim)]
        r2_list = [r2_score(y_test[:, i], xgb_preds[:, i]) for i in range(target_dim)]
        nmae_list = [float(np.mean(np.abs(y_test[:, i] - xgb_preds[:, i]))) for i in range(target_dim)]
    else:
        if use_gpu is None:
            try:
                use_gpu = torch.cuda.is_available()
            except Exception:
                use_gpu = False

        base_params = {
            "objective": "reg:squarederror",
            "max_depth": 3,
            "eta": 0.30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1.0,
            "alpha": 0.0,
            "gamma": 0.1,
            "min_child_weight": 1.0,
            "seed": 42,
        }
        if xgb_params:
            params_copy = dict(xgb_params)
            if "learning_rate" in params_copy and "eta" not in params_copy:
                params_copy["eta"] = params_copy.pop("learning_rate")
            base_params.update(params_copy)

        if feature_names is not None and len(feature_names) == int(X_train.shape[1]):
            rem_mask = np.ones((int(X_train.shape[1])), dtype=bool)

        X_train_rem = X_train[:, rem_mask]
        X_val_rem = X_val[:, rem_mask]
        X_test_rem = X_test[:, rem_mask]

        xgb_models = []
        for i in range(target_dim):
            if i == 0:
                dtrain = xgb.DMatrix(X_train_rem, label=y_train[:, i])
                dvalid = xgb.DMatrix(X_val_rem, label=y_val[:, i])
                dtest_i = xgb.DMatrix(X_test_rem)
            else:
                dtrain = xgb.DMatrix(X_train, label=y_train[:, i])
                dvalid = xgb.DMatrix(X_val, label=y_val[:, i])
                dtest_i = xgb.DMatrix(X_test)

            params = dict(base_params)
            params["tree_method"] = "gpu_hist" if use_gpu else "hist"
            try:
                booster = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=64,
                    evals=[(dvalid, f"valid_target_{i}")],
                    early_stopping_rounds=20,
                    verbose_eval=False
                )
            except xgb.core.XGBoostError:
                if params.get("tree_method") == "gpu_hist":
                    params["tree_method"] = "hist"
                    booster = xgb.train(
                        params=params,
                        dtrain=dtrain,
                        num_boost_round=64,
                        evals=[(dvalid, f"valid_target_{i}")],
                        early_stopping_rounds=20,
                        verbose_eval=False
                    )
                else:
                    raise
            xgb_models.append(booster)

        dtest_full = xgb.DMatrix(X_test)
        dtest_rem = xgb.DMatrix(X_test_rem)
        start = time.perf_counter()
        preds_list = []
        for i, booster in enumerate(xgb_models):
            dtest_i = dtest_rem if i == 0 else dtest_full
            preds_list.append(booster.predict(dtest_i))
        xgb_preds = np.column_stack(preds_list).astype(np.float32)
        end = time.perf_counter()
        xgb_total_ms = (end - start) * 1000.0
        xgb_ms_per_sample = xgb_total_ms / max(X_test.shape[0], 1)

        mse_list = [mean_squared_error(y_test[:, i], xgb_preds[:, i]) for i in range(target_dim)]
        r2_list = [r2_score(y_test[:, i], xgb_preds[:, i]) for i in range(target_dim)]
        nmae_list = [float(np.mean(np.abs(y_test[:, i] - xgb_preds[:, i]))) for i in range(target_dim)]

        if xgb_model_path:
            os.makedirs(os.path.dirname(xgb_model_path), exist_ok=True)
            for i, booster in enumerate(xgb_models):
                model_file_i = f"{xgb_model_path}_target_{i}.model"
                booster.save_model(model_file_i)
            if verbose:
                print(f"Saved XGBoost models to: {xgb_model_path}_target_{{i}}.model")

    if verbose:
        print("XGBoost Test Results:")
        if target_dim == 2:
            print(f"rem_time - MSE: {mse_list[0]:.6f}, R²: {r2_list[0]:.4f}")
            print(f"next_insn - MSE: {mse_list[1]:.6f}, R²: {r2_list[1]:.4f}")
        else:
            for i in range(target_dim):
                print(f"target_{i} - MSE: {mse_list[i]:.6f}, R²: {r2_list[i]:.4f}")
        print(f"Inference runtime: {xgb_total_ms:.3f} ms for {X_test.shape[0]} samples "
              f"({xgb_ms_per_sample:.6f} ms/sample)")

    # Fixed-point export for C inference (Q16.16)
    SCALE = 1 << 16

    def _export_fixed_point_ensemble(booster: xgb.Booster, feature_index_map: dict[int, int], out_path: str, base_q: int, input_dim: int, feature_order: list[str] | None = None):
        dumps = booster.get_dump(dump_format='json')
        with open(out_path, 'wb') as f:
            f.write(np.int32(input_dim).tobytes())
            f.write(np.int32(base_q).tobytes())
            f.write(np.uint32(len(dumps)).tobytes())
            for tjson in dumps:
                tj = json.loads(tjson)
                nodes = []
                stack = [tj]
                while stack:
                    node = stack.pop()
                    nodes.append(node)
                    for ch in node.get('children', []):
                        stack.append(ch)
                nodes.sort(key=lambda n: int(n['nodeid']))
                nid_to_idx = {int(n['nodeid']): idx for idx, n in enumerate(nodes)}

                num_nodes = len(nodes)
                f.write(np.uint32(num_nodes).tobytes())
                root_idx = np.uint32(nid_to_idx[int(tj['nodeid'])])
                f.write(root_idx.tobytes())

                use_u16_child = (num_nodes <= 65535)
                use_i16_fidx = (input_dim <= 32767)

                for n in nodes:
                    if 'leaf' in n:
                        is_leaf = np.uint8(1)
                        leaf_q = np.int32(int(np.clip(int(np.rint(float(n['leaf']) * SCALE)), -(2**31), (2**31 - 1))))
                        f.write(is_leaf.tobytes())
                        f.write(leaf_q.tobytes())
                    else:
                        is_leaf = np.uint8(0)
                        raw_split = n['split']
                        fidx_raw = int(raw_split[1:]) if isinstance(raw_split, str) and raw_split.startswith('f') else int(raw_split)
                        mapped = feature_index_map.get(fidx_raw, fidx_raw)
                        thr_q = np.int32(int(np.clip(int(np.rint(float(n['split_condition']) * SCALE)), -(2**31), (2**31 - 1))))
                        yes_idx = np.uint32(nid_to_idx[int(n['yes'])])
                        no_idx = np.uint32(nid_to_idx[int(n['no'])])

                        f.write(is_leaf.tobytes())
                        if use_i16_fidx:
                            f.write(np.int16(int(mapped)).tobytes())
                        else:
                            f.write(np.int32(int(mapped)).tobytes())
                        f.write(thr_q.tobytes())
                        if use_u16_child:
                            f.write(np.uint16(int(yes_idx)).tobytes())
                            f.write(np.uint16(int(no_idx)).tobytes())
                        else:
                            f.write(yes_idx.tobytes())
                            f.write(no_idx.tobytes())
        meta = {
            "input_dim": int(input_dim),
            "base_q": int(base_q),
            "feature_order": list(feature_order) if feature_order is not None else None,
        }
        with open(out_path + '.meta.json', 'w') as mf:
            json.dump(meta, mf)

    def _get_base_q(booster: xgb.Booster) -> int:
        cfg = json.loads(booster.save_config())
        raw_base = cfg.get('learner', {}).get('learner_model_param', {}).get('base_score', 0.5)
        def _to_float_base(val):
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                s = val.strip()
                if s.startswith('[') and s.endswith(']'):
                    s = s[1:-1].strip()
                if ',' in s:
                    s = s.split(',')[0].strip()
                try:
                    return float(s)
                except Exception:
                    return 0.5
            return 0.5
        base_score_f = _to_float_base(raw_base)
        return int(np.clip(int(np.rint(base_score_f * SCALE)), -(2**31), (2**31 - 1)))

    export_dir = os.path.dirname(os.path.abspath(xgb_model_path)) if xgb_model_path else os.getcwd()
    if feature_names is not None and len(feature_names) == int(X_train.shape[1]):
        rem_map = {i: i for i in range(int(X_train.shape[1]))}
        rem_input_dim = int(X_train.shape[1])
    full_map = {i: i for i in range(int(X_train.shape[1]))}
    full_input_dim = int(X_train.shape[1])

    base0_q = _get_base_q(xgb_models[0])
    out0 = os.path.join(export_dir, 'xgb_fp_target_0.bin')
    if feature_names is not None and len(feature_names) == int(X_train.shape[1]):
        rem_feature_order = None
    _export_fixed_point_ensemble(xgb_models[0], rem_map, out0, base0_q, rem_input_dim, rem_feature_order)
    booster1 = xgb_models[1] if len(xgb_models) > 1 else xgb_models[0]
    base1_q = _get_base_q(booster1)
    out1 = os.path.join(export_dir, 'xgb_fp_target_1.bin')
    _export_fixed_point_ensemble(booster1, full_map, out1, base1_q, full_input_dim, feature_names)

    return {
        "xgboost": {
            "predictions": xgb_preds,
            "targets": y_test.astype(np.float32),
            "models": xgb_models,
            "metrics": {
                "mse": [float(m) for m in mse_list],
                "r2": [float(r) for r in r2_list],
                "nmae": nmae_list,
                "total_ms": float(xgb_total_ms),
                "ms_per_sample": float(xgb_ms_per_sample),
            },
            "fixed_point_exports": {
                "target_0": out0,
                "target_1": out1,
            }
        }
    }


def evaluate_xgboost_c(
    xgb_results: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_features: list[str] | None,
    save_dir: str,
    so_path: str,
) -> None:
    """Run fixed-point C inference, print metrics vs float XGBoost, and save results.txt."""
    try:
        lib = ctypes.CDLL(so_path)
        lib.xgb_load_ensemble.argtypes = [ctypes.c_char_p]
        lib.xgb_load_ensemble.restype = ctypes.c_void_p
        lib.xgb_free_ensemble.argtypes = [ctypes.c_void_p]
        lib.xgb_free_ensemble.restype = None
        lib.xgb_predict_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.xgb_predict_batch.restype = ctypes.c_int

        exports = xgb_results['xgboost']['fixed_point_exports']
        bin0 = exports['target_0']
        bin1 = exports['target_1']
        h0 = lib.xgb_load_ensemble(bin0.encode('utf-8'))
        h1 = lib.xgb_load_ensemble(bin1.encode('utf-8'))
        if not h0 or not h1:
            raise RuntimeError('Failed to load fixed-point ensembles')

        SCALE = 1 << 16

        # Resolve feature order for target 0 from exported meta
        rem_order = None
        try:
            with open(bin0 + '.meta.json', 'r') as mf:
                meta0 = json.load(mf)
                if isinstance(meta0.get('feature_order'), list):
                    rem_order = [str(x) for x in meta0['feature_order']]
        except Exception:
            pass
        if rem_order is not None and input_features is not None:
            name_to_idx = {str(n): i for i, n in enumerate(input_features)}
            X0 = X_test[:, [name_to_idx[nm] for nm in rem_order]]
        else:
            rem_mask = np.ones((X_test.shape[1],), dtype=bool)
            X0 = X_test[:, rem_mask]
        X1 = X_test

        Int32P = ctypes.POINTER(ctypes.c_int32)

        def _quantize(arr):
            return np.ascontiguousarray(
                (arr.astype(np.float64) * SCALE).round().clip(-(2**31), 2**31 - 1).astype(np.int32)
            )

        X0_q = _quantize(X0)
        X1_q = _quantize(X1)
        out0_q = np.empty((X0_q.shape[0],), dtype=np.int32)
        out1_q = np.empty((X1_q.shape[0],), dtype=np.int32)

        # Full-dataset pass for NMAE (not used for timing)
        rc0 = lib.xgb_predict_batch(h0, X0_q.ctypes.data_as(Int32P), X0_q.shape[1], X0_q.shape[0], out0_q.ctypes.data_as(Int32P))
        rc1 = lib.xgb_predict_batch(h1, X1_q.ctypes.data_as(Int32P), X1_q.shape[1], X1_q.shape[0], out1_q.ctypes.data_as(Int32P))
        if rc0 != 0 or rc1 != 0:
            raise RuntimeError('C xgb_predict_batch returned error')

        preds0 = (out0_q.astype(np.float64) / SCALE).astype(np.float32)
        preds1 = (out1_q.astype(np.float64) / SCALE).astype(np.float32)
        cfp_nmae0 = float(np.mean(np.abs(y_test[:, 0] - preds0)))
        cfp_nmae1 = float(np.mean(np.abs(y_test[:, 1] - preds1)))

        # Warmup then timed repeats at batch size 20 (emulates kernel usage)
        batch_n = 20
        b0 = min(batch_n, X0_q.shape[0])
        b1 = min(batch_n, X1_q.shape[0])
        X0_b = np.ascontiguousarray(X0_q[:b0])
        X1_b = np.ascontiguousarray(X1_q[:b1])
        out0_b = np.empty((b0,), dtype=np.int32)
        out1_b = np.empty((b1,), dtype=np.int32)

        lib.xgb_predict_batch(h0, X0_b.ctypes.data_as(Int32P), X0_b.shape[1], b0, out0_b.ctypes.data_as(Int32P))
        lib.xgb_predict_batch(h1, X1_b.ctypes.data_as(Int32P), X1_b.shape[1], b1, out1_b.ctypes.data_as(Int32P))

        repeats = 10
        times0_ms, times1_ms = [], []
        for _ in range(repeats):
            t0 = time.perf_counter()
            lib.xgb_predict_batch(h0, X0_b.ctypes.data_as(Int32P), X0_b.shape[1], b0, out0_b.ctypes.data_as(Int32P))
            t1 = time.perf_counter()
            lib.xgb_predict_batch(h1, X1_b.ctypes.data_as(Int32P), X1_b.shape[1], b1, out1_b.ctypes.data_as(Int32P))
            t2 = time.perf_counter()
            times0_ms.append((t1 - t0) * 1000.0)
            times1_ms.append((t2 - t1) * 1000.0)

        ps_us_0 = (float(np.median(times0_ms)) / max(b0, 1)) * 1000.0
        ps_us_1 = (float(np.median(times1_ms)) / max(b1, 1)) * 1000.0

        lib.xgb_free_ensemble(h0)
        lib.xgb_free_ensemble(h1)

        float_nmae0 = float(xgb_results['xgboost']['metrics']['nmae'][0])
        float_nmae1 = float(xgb_results['xgboost']['metrics']['nmae'][1])
        float_r20   = float(xgb_results['xgboost']['metrics']['r2'][0])
        float_r21   = float(xgb_results['xgboost']['metrics']['r2'][1])

        def _human_size(n):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if n < 1024.0:
                    return f"{n:.2f} {unit}"
                n /= 1024.0
            return f"{n:.2f} TB"

        headers = ["Target", "NMAE (float)", "R² (float)", "NMAE (FP C)",
                   "ΔNMAE (float → FP C)", "Median Pred. Time (FP C)", "Model Size (FP C)"]
        rows = [
            ["rem_time",  f"{float_nmae0:.6f}", f"{float_r20:.4f}", f"{cfp_nmae0:.6f}",
             f"{cfp_nmae0 - float_nmae0:+.6f}", f"{ps_us_0:.2f} us/sample", _human_size(os.path.getsize(bin0))],
            ["next_insn", f"{float_nmae1:.6f}", f"{float_r21:.4f}", f"{cfp_nmae1:.6f}",
             f"{cfp_nmae1 - float_nmae1:+.6f}", f"{ps_us_1:.2f} us/sample", _human_size(os.path.getsize(bin1))],
        ]
        col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
        fmt_row = lambda cells: " | ".join(str(c).ljust(col_w[i]) for i, c in enumerate(cells))
        sep = "-+-".join("-" * w for w in col_w)
        table_lines = ["", fmt_row(headers), sep] + [fmt_row(r) for r in rows] + [""]
        for line in table_lines:
            print(line)
        results_path = os.path.join(save_dir, 'results.txt')
        with open(results_path, 'w') as f:
            f.write("\n".join(table_lines) + "\n")
        print(f"✓ Saved results table to: {results_path}")
    except Exception as e:
        print(f"Skipping C XGBoost inference due to error: {e}")

from train_utils import *
from plotting_utils import plot_xgb_nmae_histogram
import argparse
import numpy as np


def main(train_split, input_features, raw_data_file_path, normalized_data_file_path, plot_nmae_histogram=False, plot_train_runs=False, task_name=None, task_type=None, delta_tag=None):

    if task_name is None:
        task_name = os.path.basename(os.path.dirname(raw_data_file_path))

    subpath = os.path.join(task_type, task_name, delta_tag) if delta_tag else os.path.join(task_type, task_name)
    models_dir  = os.path.join('models',  subpath)
    results_dir = os.path.join('results', subpath)
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Models -> {models_dir}")
    print(f"Results -> {results_dir}")

    X_train, X_val, X_test, y_train, y_val, y_test, split, next_insn_max, rem_time_max = load_and_prepare_data(task_name, input_features, raw_data_file_path, normalized_data_file_path, random_split=False, train_split=train_split, plot_train_runs=plot_train_runs, delta_tag=delta_tag)

    print("\n" + "="*50)
    print(f"{task_name} XGBOOST MODEL TRAINING AND EVALUATION")
    print("="*50)

    xgb_model_path = os.path.join(models_dir, 'xgboost_model.json')
    xgb_results = train_and_evaluate_xgboost(
        xgb_model_path,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        verbose=False,
        feature_names=input_features)

    if plot_nmae_histogram:
        xgb_preds = xgb_results['xgboost']['predictions']
        plot_xgb_nmae_histogram(y_test, xgb_preds, results_dir)

    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c_implementation/xgb_utils_userspace.so')
    evaluate_xgboost_c(xgb_results, X_test, y_test, input_features, results_dir, so_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate XGBoost model.")
    parser.add_argument("--task", "-t", default="mcf",
                        help="Task name; used to derive default CSV path if --csv is not provided.")
    parser.add_argument("--csv", "--csv-file", dest="csv_file_path", default=None,
                        help="Path to normalized_data.csv. Defaults to profiles/{task}/normalized_data.csv.")
    parser.add_argument("--task-type", default="spec",
                        help="Benchmark suite of task.")
    parser.add_argument("--input-features", type=str, default=None,
                        help="Comma-separated list of input features to use.")
    parser.add_argument("--train-split", type=float, default=0.1,
                        help="Fraction of inputs to use for training.")
    parser.add_argument("--window-size-delta", type=float, default=0.01,
                        help="Window size delta used during preprocessing (default: 0.01).")
    parser.add_argument("--plot-nmae-histogram", action="store_true",
                        help="Plot NMAE histogram.")
    parser.add_argument("--plot-train-runs", action="store_true",
                        help="Plot training run profiles.")

    args = parser.parse_args()

    task = args.task
    delta_tag = f"delta={args.window_size_delta}"
    raw_data_file_path = args.csv_file_path or f'profiles/{task}/all_data_{delta_tag}.csv'
    normalized_data_file_path = args.csv_file_path or f'profiles/{task}/normalized_data_{delta_tag}.csv'
    input_features = args.input_features.split(',') if args.input_features else None

    main(args.train_split, input_features, raw_data_file_path, normalized_data_file_path,
         args.plot_nmae_histogram, args.plot_train_runs, task, args.task_type, delta_tag)

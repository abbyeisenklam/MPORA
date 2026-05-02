import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12


def _is_static_minmax_budget_run(run_df, min_budget, max_budget):
    if run_df['prev_budget'].nunique() != 1:
        return False
    run_budget = float(run_df['prev_budget'].iloc[0])
    return np.isclose(run_budget, min_budget) or np.isclose(run_budget, max_budget)


def plot_feature_vs_target(feature_name, target_name, x, y, point_labels=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5)
    if point_labels is not None:
        for xi, yi, label in zip(x, y, point_labels):
            plt.annotate(str(label), (xi, yi), fontsize=7, alpha=0.8)
    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.title(f"{target_name} vs. {feature_name}")
    plt.grid(True)
    plot_path = f"{target_name}_vs_{feature_name}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot: {plot_path}")


def plot_inp_feat_vs_targets(df, input_features):
    """For each inp_feat* input feature, scatter-plot it against rem_time and insn_sum."""
    agg_kwargs = {
        'prev_budget_nunique': ('prev_budget', 'nunique'),
        'inp_name_nunique': ('inp_name', 'nunique'),
        'inp_name_first': ('inp_name', 'first'),
        'rem_time_max': ('rem_time', 'max'),
        'insn_sum_max': ('insn_sum', 'max'),
    }
    for feature_name in input_features:
        if feature_name in df.columns:
            agg_kwargs[f'{feature_name}_first'] = (feature_name, 'first')
    g = df.groupby('run_id').agg(**agg_kwargs)
    g = g[(g['prev_budget_nunique'] == 1) & (g['inp_name_nunique'] == 1)]
    g = (
        g.reset_index()
         .sort_values('run_id')
         .drop_duplicates(subset=['inp_name_first'], keep='first')
         .set_index('run_id')
    )
    inp_names = g['inp_name_first'].astype(str).tolist()
    rem_times = g['rem_time_max'].tolist()
    insn_sums = g['insn_sum_max'].tolist()
    feature_columns = [
        f for f in input_features
        if f.startswith('inp_feat')
        and f[len('inp_feat'):].isdigit()
        and f'{f}_first' in g.columns
    ]
    for feature_name in feature_columns:
        feature_values = g[f'{feature_name}_first'].tolist()
        plot_feature_vs_target(feature_name, "rem_time", feature_values, rem_times, inp_names)
        plot_feature_vs_target(feature_name, "insn_sum", feature_values, insn_sums, inp_names)


def plot_train_run_profiles(task, df, train_inp_names):
    """For each training input, plot all runs colored by static vs dynamic budget."""
    print(f"Plotting profiles for train inputs: {train_inp_names}")
    budget_values = df['prev_budget'].astype(float)
    min_budget = float(budget_values.min())
    max_budget = float(budget_values.max())
    for sample_inp in train_inp_names:
        sample_df = df[df['inp_name'] == sample_inp]
        plt.figure(figsize=(10, 4))
        for run_id in sample_df['run_id'].unique():
            run_df = sample_df[sample_df['run_id'] == run_id]
            if _is_static_minmax_budget_run(run_df, min_budget, max_budget):
                plt.plot(run_df['insn_sum'], run_df['next_insn'], color='black')
            else:
                plt.plot(run_df['insn_sum'], run_df['next_insn'], alpha=0.3)
        plt.title(
            f'Executions of task {task} running with {sample_inp}',
            fontsize=16,
            fontweight='bold'
        )
        plt.xlabel('Normalized cumulative instruction count', fontsize=13)
        plt.ylabel('Normalized instruction rate', fontsize=13)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='static resource allocation (min/max budget)'),
            Line2D([0], [0], color='C0', lw=2, alpha=0.3, label='dynamic resource allocation (random walks)'),
        ]
        plt.legend(handles=legend_elements)
        plt.grid()
        plot_path = f'profiles/{task}/{sample_inp}_normalized_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_comparison_inputs(task, df, comparison_inputs):
    """Overlay runs for a list of input names, colored per input, static runs in black."""
    budget_values = df['prev_budget'].astype(float)
    min_budget = float(budget_values.min())
    max_budget = float(budget_values.max())
    available_inputs = [inp for inp in comparison_inputs if inp in df['inp_name'].astype(str).unique()]
    missing_inputs = [inp for inp in comparison_inputs if inp not in available_inputs]
    if not available_inputs:
        print("[plot_comparison_inputs] Skipping: none of the requested inputs were found.")
        return
    color_cycle = [f'C{i}' for i in range(10)]
    color_map = {inp: color_cycle[i % len(color_cycle)] for i, inp in enumerate(available_inputs)}
    plt.figure(figsize=(10, 4))
    for sample_inp in available_inputs:
        sample_df = df[df['inp_name'] == sample_inp]
        for run_id in sample_df['run_id'].unique():
            run_df = sample_df[sample_df['run_id'] == run_id]
            if _is_static_minmax_budget_run(run_df, min_budget, max_budget):
                plt.plot(run_df['insn_sum'], run_df['next_insn'], color='black', linewidth=2.0, alpha=0.9)
            else:
                plt.plot(run_df['insn_sum'], run_df['next_insn'], color=color_map[sample_inp], alpha=0.2)
    plt.title(
        f'Executions of task {task} for selected inputs',
        fontsize=16,
        fontweight='bold'
    )
    plt.xlabel('Normalized cumulative instruction count', fontsize=13)
    plt.ylabel('Normalized instruction rate', fontsize=13)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    legend_elements = [
        Line2D([0], [0], color=color_map[inp], lw=2, label=inp)
        for inp in available_inputs
    ]
    plt.legend(handles=legend_elements, ncol=min(4, max(1, len(available_inputs))))
    plt.grid()
    safe_name = '_'.join(available_inputs[:5])
    suffix = '' if len(available_inputs) <= 5 else '_plus_more'
    comparison_plot_path = f'profiles/{task}/{safe_name}{suffix}_normalized_plot.png'
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined plot for selected inputs to: {comparison_plot_path}")
    if missing_inputs:
        print(f"[plot_comparison_inputs] Inputs not found and skipped: {missing_inputs}")


def plot_two_inputs_side_by_side(task, df, inp_name_1, inp_name_2, save_path=None):
    """For two input names, plot the mean profile at min and max budget with shading between."""
    budget_values = df['prev_budget'].astype(float)
    min_budget = float(budget_values.min())
    max_budget = float(budget_values.max())

    x_grid = np.linspace(0, 1, 300)

    def _mean_profile(inp_name, target_budget):
        sample_df = df[df['inp_name'] == inp_name]
        curves = []
        for run_id in sample_df['run_id'].unique():
            run_df = sample_df[sample_df['run_id'] == run_id].sort_values('insn_sum')
            if run_df['prev_budget'].nunique() != 1:
                continue
            if not np.isclose(float(run_df['prev_budget'].iloc[0]), target_budget):
                continue
            y_interp = np.interp(x_grid, run_df['insn_sum'].values, run_df['next_insn'].values)
            curves.append(y_interp)
        return np.mean(curves, axis=0) if curves else None

    color_min = '#d6604d'
    color_max = "#043C3C"
    color_fill = 'gray'

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    for ax, inp_name in zip(axes, [inp_name_1, inp_name_2]):
        if df[df['inp_name'] == inp_name].empty:
            ax.text(0.5, 0.5, f"'{inp_name}' not found in dataframe",
                    ha='center', va='center', transform=ax.transAxes)
            continue

        mean_min = _mean_profile(inp_name, min_budget)
        mean_max = _mean_profile(inp_name, max_budget)

        if mean_min is not None and mean_max is not None:
            ax.fill_between(x_grid, mean_min, mean_max, color=color_fill, alpha=0.3)
        if mean_max is not None:
            ax.plot(x_grid, mean_max, color=color_max, lw=2)
        if mean_min is not None:
            ax.plot(x_grid, mean_min, color=color_min, lw=2)

        ax.set_xlabel('Cumulative Instruction Count', fontsize=24)
        ax.set_ylabel('Instruction Rate', fontsize=24)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True)

    legend_elements = [
        Line2D([0], [0], color=color_max, lw=2, label='max. static allocation'),
        Line2D([0], [0], color=color_min, lw=2, label='min. static allocation'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=20)

    plt.tight_layout()

    if save_path is None:
        safe_1 = inp_name_1.replace('/', '_')
        safe_2 = inp_name_2.replace('/', '_')
        save_path = f'profiles/{task}/{safe_1}_vs_{safe_2}_side_by_side.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved side-by-side plot to: {save_path}")


def plot_static_profiles(task, df, inp_names, save_dir=None):
    """For each input in inp_names, plot only constant-budget runs, colored by budget value."""
    if save_dir is None:
        save_dir = f'static/{task}'
    os.makedirs(save_dir, exist_ok=True)

    color_cycle = [f'C{i}' for i in range(10)]

    for inp_name in inp_names:
        sample_df = df[df['inp_name'] == inp_name]
        if sample_df.empty:
            print(f"[plot_static_profiles] '{inp_name}' not found, skipping.")
            continue

        static_runs = []
        for run_id in sample_df['run_id'].unique():
            run_df = sample_df[sample_df['run_id'] == run_id]
            if run_df['prev_budget'].nunique() == 1:
                budget_val = float(run_df['prev_budget'].iloc[0])
                static_runs.append((run_df, budget_val))

        if not static_runs:
            print(f"[plot_static_profiles] No static-budget runs found for '{inp_name}', skipping.")
            continue

        unique_budgets = sorted(set(b for _, b in static_runs))
        budget_color = {b: color_cycle[i % len(color_cycle)] for i, b in enumerate(unique_budgets)}

        plt.figure(figsize=(10, 4))
        for run_df, budget_val in static_runs:
            plt.plot(run_df['insn_sum'], run_df['next_insn'],
                     color=budget_color[budget_val], alpha=0.7)

        plt.title(
            f'Static-budget executions of task {task} with {inp_name}',
            fontsize=16, fontweight='bold'
        )
        plt.xlabel('Normalized cumulative instruction count', fontsize=13)
        plt.ylabel('Normalized instruction rate', fontsize=13)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        legend_elements = [
            Line2D([0], [0], color=budget_color[b], lw=2, label=f'budget={b}')
            for b in unique_budgets
        ]
        plt.legend(handles=legend_elements, ncol=min(4, max(1, len(unique_budgets))))
        plt.grid()
        plot_path = os.path.join(save_dir, f'{inp_name}_static_profiles.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved static profile plot for '{inp_name}' to: {plot_path}")


def plot_xgb_nmae_histogram(y_test, xgb_preds, save_dir):
    errors_rem  = np.abs(y_test[:, 0] - xgb_preds[:, 0])
    errors_next = np.abs(y_test[:, 1] - xgb_preds[:, 1])
    x_max = float(max(errors_rem.max(), errors_next.max()))
    n_bins = 40
    bins = np.linspace(0, x_max, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bar_w = (bins[1] - bins[0]) * 0.45
    counts_rem,  _ = np.histogram(errors_rem,  bins=bins)
    counts_next, _ = np.histogram(errors_next, bins=bins)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(bin_centers - bar_w / 2, counts_next, width=bar_w, color='#043C3C', alpha=0.8, label=r'$\mu$')
    ax.bar(bin_centers + bar_w / 2, counts_rem,  width=bar_w, color='#d62728', alpha=0.5, label=r'$e$')
    ax.set_xlim(bins[0] - bar_w, 0.6)
    ax.set_xlabel('Absolute Error (normalized)', fontsize=18)
    ax.set_ylabel('Count', fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=22, loc='upper right')
    ax.grid(True, axis='y')
    plt.tight_layout()
    mae_hist_path = os.path.join(save_dir, 'xgb_mae_histogram.pdf')
    plt.savefig(mae_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved MAE histogram to: {mae_hist_path}")

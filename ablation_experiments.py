import argparse
import importlib.util
from pathlib import Path
import time
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

from CCM import calculate_metrics
from Function import classifier_method

DEFAULT_MODES = {
    "baseline_pso": {"ql_mode": "baseline"},
    "pso_dim": {"ql_mode": "dim_only"},
    "pso_dim_random": {"ql_mode": "random_policy"},
    "full_qldgs": {"ql_mode": "full"},
}


def map_labels_hungarian(true_labels, cluster_labels):
    unique_true = np.unique(true_labels)
    unique_cluster = np.unique(cluster_labels)
    cost_matrix = np.zeros((len(unique_true), len(unique_cluster)))

    for i, true in enumerate(unique_true):
        for j, cluster in enumerate(unique_cluster):
            mask = cluster_labels == cluster
            true_labels_cluster = true_labels[mask]
            cost_matrix[i, j] = -np.sum(true_labels_cluster == true)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return {
        cluster: true for cluster, true in zip(unique_cluster[col_ind], unique_true[row_ind])
    }


def load_qldgs_module():
    module_path = Path(__file__).resolve().parent / "Method-combination" / "QLDGS-PSO-ablation.py"
    spec = importlib.util.spec_from_file_location("qldgs_pso_elite", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_datasets(dataset_names):
    dataset_dir = Path("Dataset")
    if dataset_names:
        return dataset_names
    return sorted([p.stem for p in dataset_dir.glob("*.csv")])


def load_dataset(name):
    path = Path("Dataset") / f"{name}.csv"
    data = pd.read_csv(path)
    X = data.iloc[:, :-1].values.copy()
    Y = data.iloc[:, -1].values.copy()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.nan_to_num(X)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    return X, Y


def split_main_exvute(X, Y, run, test_size):
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=run
    )
    X_test = X_valid.copy()
    Y_test = Y_valid.copy()
    return X_train, X_valid, Y_train, Y_valid, X_test, Y_test


def evaluate_single_run(qldgs, X, Y, opts, run, test_size):
    dim = X.shape[1]
    X_train, X_valid, Y_train, Y_valid, X_test, Y_test = split_main_exvute(
        X, Y, run, test_size
    )
    local_opts = opts.copy()
    local_opts["random_seed"] = int(time.time())
    start = time.time()
    FS = qldgs.fs(X_train.copy(), X_valid.copy(), Y_train.copy(), Y_valid.copy(), local_opts)
    elapsed = time.time() - start
    selected = FS["sf"] == 1
    if not np.any(selected):
        selected = np.ones(dim, dtype=bool)
    run_metrics = {"nf": FS["nf"], "time": elapsed}
    if local_opts.get("classify") == "kmeans":
        pred_valid = classifier_method(
            X_train[:, selected], X_valid[:, selected], Y_train, Y_valid, local_opts
        )
        pred_test = classifier_method(
            X_train[:, selected], X_test[:, selected], Y_train, Y_valid, local_opts
        )
        y_all = np.concatenate((Y_train, Y_valid))
        pred_valid_map = map_labels_hungarian(y_all, pred_valid)
        pred_valid = np.array([pred_valid_map[label] for label in pred_valid])
        pred_test_map = map_labels_hungarian(y_all, pred_test)
        pred_test = np.array([pred_test_map[label] for label in pred_test])
        target_valid = y_all
        target_test = y_all
    else:
        pred_valid = classifier_method(
            X_train[:, selected], X_valid[:, selected], Y_train, Y_valid, local_opts
        )
        pred_test = classifier_method(
            X_train[:, selected], X_test[:, selected], Y_train, Y_valid, local_opts
        )
        target_valid = Y_valid
        target_test = Y_test

    acc_valid = calculate_metrics(target_valid, pred_valid, metric="accuracy")
    acc_test = calculate_metrics(target_test, pred_test, metric="accuracy")
    f1_valid_metrics = calculate_metrics(target_valid, pred_valid, metric="f1")
    f1_test_metrics = calculate_metrics(target_test, pred_test, metric="f1")
    auc_valid_metrics = calculate_metrics(target_valid, pred_valid, metric="roc_auc")
    auc_test_metrics = calculate_metrics(target_test, pred_test, metric="roc_auc")
    recall_valid_metrics = calculate_metrics(target_valid, pred_valid, metric="recall")
    recall_test_metrics = calculate_metrics(target_test, pred_test, metric="recall")
    precision_valid_metrics = calculate_metrics(target_valid, pred_valid, metric="precision")
    precision_test_metrics = calculate_metrics(target_test, pred_test, metric="precision")
    run_metrics.update(
        {
            "acc_valid": acc_valid,
            "acc_test": acc_test,
            "f1_valid": f1_valid_metrics["F1Score (Macro)"],
            "f1_test": f1_test_metrics["F1Score (Macro)"],
            "auc_valid": auc_valid_metrics["ROC AUC (Macro)"],
            "auc_test": auc_test_metrics["ROC AUC (Macro)"],
            "recall_valid": recall_valid_metrics["Recall (Macro)"],
            "recall_test": recall_test_metrics["Recall (Macro)"],
            "precision_valid": precision_valid_metrics["Precision (Macro)"],
            "precision_test": precision_test_metrics["Precision (Macro)"],
        }
    )
    return run_metrics


def _evaluate_single_run_subprocess(args):
    run, X, Y, opts, test_size = args
    qldgs = load_qldgs_module()
    return run, evaluate_single_run(qldgs, X, Y, opts, run, test_size)


def evaluate_configuration(qldgs, X, Y, opts, runs, test_size, workers):
    worker_count = workers or (os.cpu_count() or 1)
    worker_count = max(1, min(worker_count, runs))
    if worker_count == 1:
        return [
            evaluate_single_run(qldgs, X, Y, opts, run, test_size) for run in range(runs)
        ]

    results = [None] * runs
    parallel_outputs = Parallel(n_jobs=worker_count, prefer="processes")(
        delayed(_evaluate_single_run_subprocess)((run, X, Y, opts, test_size))
        for run in range(runs)
    )
    for run_idx, metrics in parallel_outputs:
        results[run_idx] = metrics
    return results


def run(args):
    datasets = discover_datasets(args.datasets)
    if not datasets:
        raise RuntimeError("No datasets found for ablation study")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    if args.split_mode != "main_exvute":
        raise ValueError("Ablation now shares Main_exvute's evaluation; set --split-mode main_exvute.")
    qldgs = load_qldgs_module()
    selected_modes = args.modes or list(DEFAULT_MODES.keys())
    records = []
    for dataset in datasets:
        X, Y = load_dataset(dataset)
        for mode in selected_modes:
            if mode not in DEFAULT_MODES:
                raise ValueError(f"Unknown mode '{mode}'. Available: {list(DEFAULT_MODES.keys())}")
            mode_cfg = DEFAULT_MODES[mode]
            local_opts = {
                "N": args.population,
                "T": args.iterations,
                "ql_lr": args.ql_lr,
                "ql_gamma": args.ql_gamma,
                "interval_num": args.interval_num,
                "interval_iterations": args.interval_iterations,
                "phi": args.phi,
                "cluster": ",",
                "classify": args.classifier,
                "split": args.split,
                "func": args.func,
                "knn_para": args.knn_k,
            }
            local_opts.update(mode_cfg)
            print(f"{dataset} -> {mode}")
            mode_results = evaluate_configuration(
                qldgs,
                X,
                Y,
                local_opts,
                args.runs,
                args.test_size,
                args.workers,
            )
            for run_idx, metrics in enumerate(mode_results):
                record = {
                    "dataset": dataset,
                    "mode": mode,
                    "run": run_idx,
                    "ql_mode": mode_cfg.get("ql_mode"),
                    "nf": metrics["nf"],
                    "time": metrics["time"],
                    "acc_test": metrics["acc_test"],
                    "f1_test": metrics["f1_test"],
                    "auc_test": metrics["auc_test"],
                }
                record.update(
                    {
                        "acc_valid": metrics["acc_valid"],
                        "f1_valid": metrics["f1_valid"],
                        "auc_valid": metrics["auc_valid"],
                        "recall_valid": metrics["recall_valid"],
                        "recall_test": metrics["recall_test"],
                        "precision_valid": metrics["precision_valid"],
                        "precision_test": metrics["precision_test"],
                    }
                )
                records.append(record)
    df = pd.DataFrame(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    agg_kwargs = {
        "acc_test_mean": ("acc_test", "mean"),
        "acc_test_std": ("acc_test", "std"),
        "f1_test_mean": ("f1_test", "mean"),
        "f1_test_std": ("f1_test", "std"),
        "auc_test_mean": ("auc_test", "mean"),
        "auc_test_std": ("auc_test", "std"),
        "recall_test_mean": ("recall_test", "mean"),
        "recall_test_std": ("recall_test", "std"),
        "precision_test_mean": ("precision_test", "mean"),
        "precision_test_std": ("precision_test", "std"),
        "nf_mean": ("nf", "mean"),
        "nf_std": ("nf", "std"),
        "time_mean": ("time", "mean"),
        "acc_valid_mean": ("acc_valid", "mean"),
        "acc_valid_std": ("acc_valid", "std"),
        "f1_valid_mean": ("f1_valid", "mean"),
        "f1_valid_std": ("f1_valid", "std"),
        "auc_valid_mean": ("auc_valid", "mean"),
        "auc_valid_std": ("auc_valid", "std"),
        "recall_valid_mean": ("recall_valid", "mean"),
        "recall_valid_std": ("recall_valid", "std"),
        "precision_valid_mean": ("precision_valid", "mean"),
        "precision_valid_std": ("precision_valid", "std"),
    }
    summary = df.groupby(["dataset", "mode"]).agg(**agg_kwargs).reset_index()
    summary_path = output.with_name(f"{output.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Ablation raw logs saved to {output}")
    print(f"Ablation summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="QLDGS-PSO-Elite ablation runner")
    parser.add_argument("--datasets", nargs="*", help="Dataset names without .csv extension")
    parser.add_argument("--runs", type=int, default=5, help="Independent runs per mode")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes for parallel runs (0 uses available CPU cores)",
    )
    parser.add_argument("--population", type=int, default=20, help="Population size N")
    parser.add_argument("--iterations", type=int, default=500, help="Iterations per run T")
    parser.add_argument("--interval-num", type=int, default=20, help="Number of intervals L")
    parser.add_argument(
        "--interval-iterations", type=int, default=20, help="Iterations per interval M"
    )
    parser.add_argument("--ql-lr", type=float, default=0.2, help="Q-learning learning rate lr")
    parser.add_argument("--ql-gamma", type=float, default=0.5, help="Q-learning discount γ")
    parser.add_argument("--phi", type=float, default=0.99, help="Fitness weighting φ")
    parser.add_argument("--classifier", type=str, default="knn", help="Classifier choice")
    parser.add_argument("--knn-k", type=int, default=3, help="k for KNN classifier")
    parser.add_argument("--split", type=int, default=0, help="Split strategy used in Fun")
    parser.add_argument("--func", type=int, default=0, help="Objective variant")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction reserved for the shared validation/test split (0.3 in Main_exvute).",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=list(DEFAULT_MODES.keys()),
        help="Subset of ablation modes to run",
    )
    parser.add_argument(
        "--split-mode",
        choices=["main_exvute"],
        default="main_exvute",
        help="Evaluation strictly mirrors Main_exvute.py (fixed 7:3 split).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Result/ablation_experiments.csv",
        help="CSV file for ablation logs",
    )
    args = parser.parse_args()
    
    if args.workers > 0:
        print("workers:", args.workers)
        # os.environ["OMP_NUM_THREADS"] = "1"
        # os.environ["MKL_NUM_THREADS"] = "1"
        # os.environ["OPENBLAS_NUM_THREADS"] = "1"
        # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    run(args)


if __name__ == "__main__":
    main()

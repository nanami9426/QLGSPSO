import argparse
import itertools
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from CCM import calculate_metrics
from Function import classifier_method


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
    module_path = Path(__file__).resolve().parent / "Method-combination" / "QLDGS-PSO-param_sens.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot locate QLDGS-PSO-Elite.py at {module_path}")
    spec = importlib.util.spec_from_file_location("qldgs_pso_elite", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_numeric_list(text, cast_func=float):
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(cast_func(item))
    return values


def discover_datasets(dataset_names):
    dataset_dir = Path("Dataset")
    if dataset_names:
        return dataset_names
    return sorted([p.stem for p in dataset_dir.glob("*.csv")])


def load_dataset(name):
    dataset_path = Path("Dataset") / f"{name}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_path} not found")
    data = pd.read_csv(dataset_path)
    X = data.iloc[:, :-1].values.copy()
    Y = data.iloc[:, -1].values.copy()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.nan_to_num(X)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    return X, Y


def split_main_exvute(X, Y, run, test_size):
    """Replicate the 7:3 split used in Main_exvute.py."""
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=run
    )
    X_test = X_valid.copy()
    Y_test = Y_valid.copy()
    return X_train, X_valid, Y_train, Y_valid, X_test, Y_test


def build_parameter_configs(param_grid, strategy):
    base_config = {k: v[0] for k, v in param_grid.items()}
    configs = []
    if strategy == "grid":
        keys = list(param_grid.keys())
        for values in itertools.product(*param_grid.values()):
            config = dict(zip(keys, values))
            config["_varying"] = "grid"
            configs.append(config)
        return configs, base_config
    # one-vs-rest: vary one parameter at a time while others stay at baseline
    for key, values in param_grid.items():
        for value in values:
            config = base_config.copy()
            config[key] = value
            config["_varying"] = key
            configs.append(config)
    return configs, base_config


def evaluate_configuration(qldgs_module, X, Y, opts, runs, test_size):
    """Run the FS method with the exact evaluation protocol from Main_exvute.py."""
    dim = X.shape[1]
    results = []
    for run in range(runs):
        X_train, X_valid, Y_train, Y_valid, X_test, Y_test = split_main_exvute(
            X, Y, run, test_size
        )
        local_opts = opts.copy()
        local_opts["random_seed"] = run
        start_time = time.time()
        FS = qldgs_module.fs(
            X_train.copy(), X_valid.copy(), Y_train.copy(), Y_valid.copy(), local_opts
        )
        elapsed = time.time() - start_time
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
        precision_valid_metrics = calculate_metrics(
            target_valid, pred_valid, metric="precision"
        )
        precision_test_metrics = calculate_metrics(
            target_test, pred_test, metric="precision"
        )
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
        results.append(run_metrics)
    return results


def run_sensitivity(args):
    datasets = discover_datasets(args.datasets)
    if not datasets:
        raise RuntimeError("No datasets found under Dataset/")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be within (0,1)")
    if args.split_mode != "main_exvute":
        raise ValueError("This script now mirrors Main_exvute.py exclusively; use --split-mode main_exvute.")
    qldgs_module = load_qldgs_module()
    param_grid = {
        "phi": parse_numeric_list(args.phi, float),
        "ql_lr": parse_numeric_list(args.ql_lr, float),
        "ql_gamma": parse_numeric_list(args.ql_gamma, float),
        "interval_num": parse_numeric_list(args.interval_num, int),
        "interval_iterations": parse_numeric_list(args.interval_iterations, int),
        "T": parse_numeric_list(args.iterations, int),
    }
    configs, base_config = build_parameter_configs(param_grid, args.strategy)
    base_opts = {
        "maxLt": args.runs,
        "N": args.population,
        "T": base_config["T"],
        "phi": base_config["phi"],
        "interval_iterations": base_config["interval_iterations"],
        "cluster": ",",
        "classify": args.classifier,
        "split": args.split,
        "func": args.func,
        "knn_para": args.knn_k,
    }
    all_records = []
    for dataset in datasets:
        X, Y = load_dataset(dataset)
        for config_id, config in enumerate(configs):
            label = f"{config.get('_varying', 'grid')}->{config_id}"
            print(f"[{dataset}] Testing {label} with settings {config}")
            local_opts = base_opts.copy()
            local_opts.update({k: v for k, v in config.items() if not k.startswith("_")})
            results = evaluate_configuration(
                qldgs_module,
                X,
                Y,
                local_opts,
                args.runs,
                args.test_size,
            )
            for run_idx, metrics in enumerate(results):
                record = {
                    "dataset": dataset,
                    "run": run_idx,
                    "varying_param": config.get("_varying", "grid"),
                    "phi": local_opts["phi"],
                    "ql_lr": local_opts["ql_lr"],
                    "ql_gamma": local_opts["ql_gamma"],
                    "interval_num": local_opts["interval_num"],
                    "interval_iterations": local_opts["interval_iterations"],
                    "iterations": local_opts["T"],
                }
                record.update(metrics)
                all_records.append(record)
    if not all_records:
        raise RuntimeError("No experimental records were produced")
    results_df = pd.DataFrame(all_records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    group_cols = [
        "dataset",
        "varying_param",
        "phi",
        "ql_lr",
        "ql_gamma",
        "interval_num",
        "interval_iterations",
        "iterations",
    ]
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
    summary = results_df.groupby(group_cols).agg(**agg_kwargs).reset_index()
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved raw records to {output_path}")
    print(f"Saved summary statistics to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parameter sensitivity sweeps for QLDGS-PSO-Elite"
    )
    parser.add_argument("--datasets", nargs="*", help="Dataset names without .csv suffix")
    parser.add_argument("--runs", type=int, default=5, help="Independent runs per setting")
    parser.add_argument("--population", type=int, default=20, help="Population size N")
    parser.add_argument(
        "--iterations", type=str, default="200,400,600", help="Comma separated T values"
    )
    parser.add_argument(
        "--interval-num",
        type=str,
        default="10,15,20",
        help="Comma separated interval counts L",
    )
    parser.add_argument(
        "--interval-iterations",
        type=str,
        default="20,30,40",
        help="Comma separated iteration counts per interval M",
    )
    parser.add_argument(
        "--ql-lr", type=str, default="0.1,0.2,0.4", help="Comma separated learning rates"
    )
    parser.add_argument(
        "--ql-gamma",
        type=str,
        default="0.0,0.3,0.6",
        help="Comma separated discount factors",
    )
    parser.add_argument(
        "--phi",
        type=str,
        default="0.95,0.98,0.99",
        help="Comma separated fitness-factor weights φ",
    )
    parser.add_argument("--classifier", type=str, default="knn", help="Classifier for evaluation")
    parser.add_argument("--knn-k", type=int, default=3, help="k used when classifier=knn")
    parser.add_argument("--split", type=int, default=0, help="Split strategy used in Fun")
    parser.add_argument("--func", type=int, default=0, help="Objective variant index")
    parser.add_argument(
        "--split-mode",
        choices=["main_exvute"],
        default="main_exvute",
        help="Evaluation strictly mirrors Main_exvute.py (fixed 7:3 split).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of samples used as held-out validation/test set (0.3 in Main_exvute).",
    )
    parser.add_argument(
        "--strategy",
        choices=["grid", "one-vs-rest"],
        default="one-vs-rest",
        help="Sweep strategy for parameter grid",
    )
    parser.add_argument(
        "--output", type=str, default="Result/parameter_sensitivity.csv", help="CSV output path"
    )
    args = parser.parse_args()
    run_sensitivity(args)


if __name__ == "__main__":
    main()

import os
import sys
import time
import datetime
import importlib
import warnings
import random

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import DataConversionWarning
from joblib import Parallel, delayed

from CCM import calculate_metrics
from Function import classifier_method
from Save import append_to_csv


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


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
    label_map = {
        cluster: true for cluster, true in zip(unique_cluster[col_ind], unique_true[row_ind])
    }

    return label_map


def map_labels(true_labels, cluster_labels):
    unique_cluster = np.unique(cluster_labels)
    label_map = {}
    for cluster in unique_cluster:
        mask = cluster_labels == cluster
        true_labels_cluster = true_labels[mask]
        counts = np.bincount(true_labels_cluster)
        max_count = np.argmax(counts)
        label_map[cluster] = max_count
    return label_map


def determine_fold_count(labels, desired_folds):
    _, counts = np.unique(labels, return_counts=True)
    min_class_samples = np.min(counts)
    if min_class_samples < 2:
        return 0, min_class_samples
    folds = min(desired_folds, int(min_class_samples))
    return folds, min_class_samples


def _run_single_fold(
    fs_method,
    fold_idx,
    train_index,
    test_index,
    X,
    Y,
    opts,
    base_seed,
    run_idx,
):
    X_train = X[train_index]
    Y_train = Y[train_index]
    X_valid = X[test_index]
    Y_valid = Y[test_index]
    X_test = X_valid
    Y_test = Y_valid
    local_opts = opts.copy()
    local_opts["random_seed"] = base_seed + run_idx * 1000 + fold_idx

    start_time = time.time()
    FS = fs_method(X_train, X_valid, Y_train, Y_valid, local_opts)
    end_time = time.time()

    selected_mask = np.asarray(FS["sf"]).reshape(-1)
    curve = np.asarray(FS["c"]).reshape(-1)
    nf = FS["nf"]
    elapsed = end_time - start_time
    if random.random() < 0.2:
        print(f"run: {run_idx}, fold: {fold_idx}, elapsed: {round(elapsed, 2)}")
    if local_opts["classify"] == "kmeans":
        pred_valid = classifier_method(
            X_train[:, selected_mask == 1],
            X_valid[:, selected_mask == 1],
            Y_train,
            Y_valid,
            local_opts,
        )
        pred_test = classifier_method(
            X_train[:, selected_mask == 1],
            X_test[:, selected_mask == 1],
            Y_train,
            Y_valid,
            local_opts,
        )
        y_all = np.concatenate((Y_train, Y_valid))
        pred_test_map = map_labels_hungarian(y_all, pred_test)
        pred_test = np.array([pred_test_map[label] for label in pred_test])
        pred_valid_map = map_labels_hungarian(y_all, pred_valid)
        pred_valid = np.array([pred_valid_map[label] for label in pred_valid])

        acc_valid = calculate_metrics(y_all, pred_valid, metric="accuracy")
        acc_test = calculate_metrics(y_all, pred_test, metric="accuracy")
        f1_valid_metrics = calculate_metrics(y_all, pred_valid, metric="f1")
        f1_valid = f1_valid_metrics["F1Score (Macro)"]
        f1_test_metrics = calculate_metrics(y_all, pred_test, metric="f1")
        f1_test = f1_test_metrics["F1Score (Macro)"]
        auc_valid_metrics = calculate_metrics(y_all, pred_valid, metric="roc_auc")
        auc_valid = auc_valid_metrics["ROC AUC (Macro)"]
        auc_test_metrics = calculate_metrics(y_all, pred_test, metric="roc_auc")
        auc_test = auc_test_metrics["ROC AUC (Macro)"]
        recall_valid_metrics = calculate_metrics(y_all, pred_valid, metric="recall")
        recall_valid = recall_valid_metrics["Recall (Macro)"]
        recall_test_metrics = calculate_metrics(y_all, pred_test, metric="recall")
        recall_test = recall_test_metrics["Recall (Macro)"]
        precision_valid_metrics = calculate_metrics(y_all, pred_valid, metric="precision")
        precision_valid = precision_valid_metrics["Precision (Macro)"]
        precision_test_metrics = calculate_metrics(y_all, pred_test, metric="precision")
        precision_test = precision_test_metrics["Precision (Macro)"]
    else:
        pred_valid = classifier_method(
            X_train[:, selected_mask == 1],
            X_valid[:, selected_mask == 1],
            Y_train,
            Y_valid,
            local_opts,
        )
        pred_test = classifier_method(
            X_train[:, selected_mask == 1],
            X_test[:, selected_mask == 1],
            Y_train,
            Y_valid,
            local_opts,
        )

        acc_valid = calculate_metrics(Y_valid, pred_valid, metric="accuracy")
        acc_test = calculate_metrics(Y_test, pred_test, metric="accuracy")
        f1_valid_metrics = calculate_metrics(Y_valid, pred_valid, metric="f1")
        f1_valid = f1_valid_metrics["F1Score (Macro)"]
        f1_test_metrics = calculate_metrics(Y_test, pred_test, metric="f1")
        f1_test = f1_test_metrics["F1Score (Macro)"]
        auc_valid_metrics = calculate_metrics(Y_valid, pred_valid, metric="roc_auc")
        auc_valid = auc_valid_metrics["ROC AUC (Macro)"]
        auc_test_metrics = calculate_metrics(Y_test, pred_test, metric="roc_auc")
        auc_test = auc_test_metrics["ROC AUC (Macro)"]
        recall_valid_metrics = calculate_metrics(Y_valid, pred_valid, metric="recall")
        recall_valid = recall_valid_metrics["Recall (Macro)"]
        recall_test_metrics = calculate_metrics(Y_test, pred_test, metric="recall")
        recall_test = recall_test_metrics["Recall (Macro)"]
        precision_valid_metrics = calculate_metrics(Y_valid, pred_valid, metric="precision")
        precision_valid = precision_valid_metrics["Precision (Macro)"]
        precision_test_metrics = calculate_metrics(Y_test, pred_test, metric="precision")
        precision_test = precision_test_metrics["Precision (Macro)"]

    return {
        "fold_idx": fold_idx,
        "sf": selected_mask,
        "nf": nf,
        "curve": curve,
        "time": elapsed,
        "acc_valid": acc_valid,
        "acc_test": acc_test,
        "f1_valid": f1_valid,
        "f1_test": f1_test,
        "auc_valid": auc_valid,
        "auc_test": auc_test,
        "recall_valid": recall_valid,
        "recall_test": recall_test,
        "precision_valid": precision_valid,
        "precision_test": precision_test,
    }


def _run_single_run(
    run_idx,
    fold_indices,
    fs_method,
    X,
    Y,
    opts,
    base_seed,
    dim,
    folds,
    n_jobs_folds,
    method,
    data_name,
):
    fold_sf = np.zeros((folds, dim))
    fold_nf = np.zeros((folds, 1))
    fold_curve = np.zeros((folds, opts["T"]))
    fold_timecal = np.zeros((folds, 1))
    fold_acc_valid = np.zeros((folds, 1))
    fold_acc_test = np.zeros((folds, 1))
    fold_f1_valid = np.zeros((folds, 1))
    fold_f1_test = np.zeros((folds, 1))
    fold_auc_valid = np.zeros((folds, 1))
    fold_auc_test = np.zeros((folds, 1))
    fold_recall_valid = np.zeros((folds, 1))
    fold_recall_test = np.zeros((folds, 1))
    fold_precision_valid = np.zeros((folds, 1))
    fold_precision_test = np.zeros((folds, 1))

    fold_results = Parallel(n_jobs=n_jobs_folds)(
        delayed(_run_single_fold)(
            fs_method,
            fold_idx,
            train_index,
            test_index,
            X,
            Y,
            opts,
            base_seed,
            run_idx,
        )
        for fold_idx, (train_index, test_index) in enumerate(fold_indices)
    )

    for res in fold_results:
        fi = res["fold_idx"]
        fold_sf[fi, :] = res["sf"]
        fold_nf[fi, 0] = res["nf"]
        fold_curve[fi, :] = res["curve"]
        fold_timecal[fi, 0] = res["time"]
        fold_acc_valid[fi, 0] = res["acc_valid"]
        fold_acc_test[fi, 0] = res["acc_test"]
        fold_f1_valid[fi, 0] = res["f1_valid"]
        fold_f1_test[fi, 0] = res["f1_test"]
        fold_auc_valid[fi, 0] = res["auc_valid"]
        fold_auc_test[fi, 0] = res["auc_test"]
        fold_recall_valid[fi, 0] = res["recall_valid"]
        fold_recall_test[fi, 0] = res["recall_test"]
        fold_precision_valid[fi, 0] = res["precision_valid"]
        fold_precision_test[fi, 0] = res["precision_test"]

    best_fold_idx = int(np.argmin(fold_curve[:, -1]))
    run_sf = fold_sf[best_fold_idx, :]
    nf_mean = np.mean(fold_nf)
    curve_mean = np.mean(fold_curve, axis=0)
    time_mean = np.mean(fold_timecal)
    acc_valid_mean = np.mean(fold_acc_valid)
    acc_test_mean = np.mean(fold_acc_test)
    f1_valid_mean = np.mean(fold_f1_valid)
    f1_test_mean = np.mean(fold_f1_test)
    auc_valid_mean = np.mean(fold_auc_valid)
    auc_test_mean = np.mean(fold_auc_test)
    recall_valid_mean = np.mean(fold_recall_valid)
    recall_test_mean = np.mean(fold_recall_test)
    precision_valid_mean = np.mean(fold_precision_valid)
    precision_test_mean = np.mean(fold_precision_test)

    run_result = {
        "Data Name": data_name,
        "Folds": folds,
        "Method": method,
        "Run": run_idx + 1,
        "Feature Number": nf_mean,
        "Time Calculation": time_mean,
        "Accuracy Valid": acc_valid_mean,
        "Accuracy Test": acc_test_mean,
        "F1 Valid": f1_valid_mean,
        "F1 Test": f1_test_mean,
        "AUC Valid": auc_valid_mean,
        "AUC Test": auc_test_mean,
        "Recall Valid": recall_valid_mean,
        "Recall Test": recall_test_mean,
        "Precision Valid": precision_valid_mean,
        "Precision Test": precision_test_mean,
    }

    return {
        "run_idx": run_idx,
        "sf": run_sf,
        "nf": nf_mean,
        "curve": curve_mean,
        "time": time_mean,
        "acc_valid": acc_valid_mean,
        "acc_test": acc_test_mean,
        "f1_valid": f1_valid_mean,
        "f1_test": f1_test_mean,
        "auc_valid": auc_valid_mean,
        "auc_test": auc_test_mean,
        "recall_valid": recall_valid_mean,
        "recall_test": recall_test_mean,
        "precision_valid": precision_valid_mean,
        "precision_test": precision_test_mean,
        "best_fold_idx": best_fold_idx,
        "run_result": run_result,
    }


def run_cross_validation(
    data_name,
    method_list,
    base_opts,
    desired_folds,
    current_date,
    runs=None,
    output_dir="Result",
    file_prefix="cv",
):
    path = "Dataset"
    data_path = os.path.join(path, f"{data_name}.csv")
    if not os.path.exists(data_path):
        print(f"未找到数据集文件：{data_path}")
        return []

    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1].values.copy()
    Y = data.iloc[:, -1].values.copy()
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    X = np.nan_to_num(X)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X)

    sample = np.size(X, 0)
    dim = np.size(X, 1)
    folds, min_class_sample = determine_fold_count(Y, desired_folds)
    if min_class_sample < 2 or folds < 2:
        print(
            f"数据集【{data_name}】存在样本量少于2的类别或无法划分足够的折数，跳过该数据集。"
        )
        return []
    if folds < desired_folds:
        print(
            f"数据集【{data_name}】最小类别样本数为{min_class_sample}，使用{folds}折交叉验证代替{desired_folds}折。"
        )

    run_repeats = runs if runs is not None else 1  # repeat full k-fold runs
    opts = base_opts.copy()
    opts["maxLt"] = run_repeats
    total_runs = run_repeats
    print(
        f"数据集【{data_name}】，样本数量：{sample}；特征数量：{dim}；折数：{folds}；重复次数：{run_repeats}"
    )

    base_seed = int(time.time())
    fold_splits_by_run = []
    for repeat_idx in range(run_repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=base_seed + repeat_idx)
        fold_splits_by_run.append(list(skf.split(X, Y)))

    method_num = len(method_list)
    result_list = []
    result_curve = []
    result_run = []
    result_run_best = np.zeros((total_runs, method_num))
    result_run_best_name = np.array([[data_name] for _ in range(total_runs)], dtype=object)
    results = {data_name: {}}
    dataset_rows = []

    for method_idx, method in enumerate(method_list):
        sf = np.zeros((total_runs, dim))
        nf = np.zeros((total_runs, 1))
        curve = np.zeros((total_runs, opts["T"]))
        timecal = np.zeros((total_runs, 1))
        acc_valid = np.zeros((total_runs, 1))
        acc_test = np.zeros((total_runs, 1))
        f1_valid = np.zeros((total_runs, 1))
        f1_test = np.zeros((total_runs, 1))
        auc_valid = np.zeros((total_runs, 1))
        auc_test = np.zeros((total_runs, 1))
        recall_valid = np.zeros((total_runs, 1))
        recall_test = np.zeros((total_runs, 1))
        precision_valid = np.zeros((total_runs, 1))
        precision_test = np.zeros((total_runs, 1))
        best_fold_per_run = []

        try:
            module = importlib.import_module(method)
        except ModuleNotFoundError as exc:
            print(f"未找到算法【{method}】：{exc}")
            continue

        fs_method = getattr(module, "fs", None)
        if fs_method is None or not callable(fs_method):
            print(f"算法【{method}】未找到可调用的 fs 函数")
            continue

        method_has_runs = False
        n_jobs_runs = opts.get("n_jobs_runs", 1)
        n_jobs_folds = opts.get("n_jobs_folds", opts.get("n_jobs", 1))
        if n_jobs_runs > 1 and "n_jobs_folds" not in opts and "n_jobs" not in opts:
            n_jobs_folds = 1  # avoid oversubscribing CPUs when nesting parallel loops

        run_tasks = list(enumerate(fold_splits_by_run))
        if n_jobs_runs == 1:
            run_outputs = []
            for run_idx, fold_indices in run_tasks:
                method_has_runs = True
                run_outputs.append(
                    _run_single_run(
                        run_idx,
                        fold_indices,
                        fs_method,
                        X,
                        Y,
                        opts,
                        base_seed,
                        dim,
                        folds,
                        n_jobs_folds,
                        method,
                        data_name,
                    )
                )
        else:
            method_has_runs = bool(run_tasks)

            run_outputs = Parallel(n_jobs=n_jobs_runs)(
                delayed(_run_single_run)(
                    run_idx,
                    fold_indices,
                    fs_method,
                    X,
                    Y,
                    opts,
                    base_seed,
                    dim,
                    folds,
                    n_jobs_folds,
                    method,
                    data_name,
                )
                for run_idx, fold_indices in run_tasks
            )

        for res in run_outputs:
            ri = res["run_idx"]
            best_fold_per_run.append(res["best_fold_idx"])
            sf[ri, :] = res["sf"]
            nf[ri, 0] = res["nf"]
            curve[ri, :] = res["curve"]
            timecal[ri, 0] = res["time"]
            acc_valid[ri, 0] = res["acc_valid"]
            acc_test[ri, 0] = res["acc_test"]
            f1_valid[ri, 0] = res["f1_valid"]
            f1_test[ri, 0] = res["f1_test"]
            auc_valid[ri, 0] = res["auc_valid"]
            auc_test[ri, 0] = res["auc_test"]
            recall_valid[ri, 0] = res["recall_valid"]
            recall_test[ri, 0] = res["recall_test"]
            precision_valid[ri, 0] = res["precision_valid"]
            precision_test[ri, 0] = res["precision_test"]
            result_run.append(res["run_result"])

        if not method_has_runs:
            continue

        best_run_idx = int(np.argmin(curve[:, -1]))
        best_fold_idx = best_fold_per_run[best_run_idx]
        sf_best = sf[best_run_idx, :]
        curve_mean = np.mean(curve, axis=0, keepdims=True)
        fitness_mean = curve_mean[0, -1]
        acc_test_best = np.max(acc_test)
        nf_mean = np.mean(nf)
        nf_std = np.std(nf)
        timecal_mean = np.mean(timecal)
        timecal_std = np.std(timecal)
        acc_test_mean = np.mean(acc_test)
        acc_test_std = np.std(acc_test)
        acc_test_min = np.min(acc_test)
        f1_test_mean = np.mean(f1_test)
        auc_test_mean = np.mean(auc_test)
        recall_test_mean = np.mean(recall_test)
        precision_test_mean = np.mean(precision_test)

        train_best_idx, valid_best_idx = fold_splits_by_run[best_run_idx][best_fold_idx]
        X_train_best = X[train_best_idx]
        Y_train_best = Y[train_best_idx]
        X_valid_best = X[valid_best_idx]
        Y_valid_best = Y[valid_best_idx]
        X_test_best = X_valid_best.copy()
        Y_test_best = Y_valid_best.copy()

        valid_pred = classifier_method(
            X_train_best[:, sf_best == 1],
            X_valid_best[:, sf_best == 1],
            Y_train_best,
            Y_valid_best,
            opts,
        )
        test_pred = classifier_method(
            X_train_best[:, sf_best == 1],
            X_test_best[:, sf_best == 1],
            Y_train_best,
            Y_valid_best,
            opts,
        )
        if opts["classify"] == "kmeans":
            y_all = np.concatenate((Y_train_best, Y_valid_best))
            pred_test_map = map_labels_hungarian(y_all, test_pred)
            test_pred = np.array([pred_test_map[label] for label in test_pred])
            pred_valid_map = map_labels_hungarian(y_all, valid_pred)
            valid_pred = np.array([pred_valid_map[label] for label in valid_pred])
            valid_result = calculate_metrics(y_all, valid_pred, "all")
            test_result = calculate_metrics(y_all, test_pred, "all")
        else:
            valid_result = calculate_metrics(Y_valid_best, valid_pred, "all")
            test_result = calculate_metrics(Y_test_best, test_pred, "all")

        best_feature_indices = np.where(sf_best == 1)[0].tolist()
        row = {
            "Dataset": data_name,
            "Samples": int(sample),
            "Features": int(dim),
            "Folds": int(folds),
            "Method": method,
            "NF Mean": float(nf_mean),
            "NF Std": float(nf_std),
            "Acc Test Best": float(acc_test_best),
            "Acc Test Mean": float(acc_test_mean),
            "Acc Test Std": float(acc_test_std),
            "Acc Test Min": float(acc_test_min),
            "F1 Test Mean": float(f1_test_mean),
            "AUC Test Mean": float(auc_test_mean),
            "Recall Test Mean": float(recall_test_mean),
            "Precision Test Mean": float(precision_test_mean),
            "Time Mean": float(timecal_mean),
            "Time Std": float(timecal_std),
            "Best Run": best_run_idx + 1,
            "Best Fold": best_fold_idx + 1,
            "Best Valid Accuracy": float(valid_result["Accuracy"]),
            "Best Valid Recall Macro": float(valid_result["Recall (Macro)"]),
            "Best Valid Precision Macro": float(valid_result["Precision (Macro)"]),
            "Best Valid F1 Macro": float(valid_result["F1-Score (Macro)"]),
            "Best Valid AUC Macro": float(valid_result["ROC AUC (Macro)"]),
            "Best Test Accuracy": float(test_result["Accuracy"]),
            "Best Test Recall Macro": float(test_result["Recall (Macro)"]),
            "Best Test Precision Macro": float(test_result["Precision (Macro)"]),
            "Best Test F1 Macro": float(test_result["F1-Score (Macro)"]),
            "Best Test AUC Macro": float(test_result["ROC AUC (Macro)"]),
            "Selected Feature Count": int(np.sum(sf_best)),
            "Selected Feature Indices": ";".join(map(str, best_feature_indices)),
        }
        dataset_rows.append(row)

        result_list.append(
            [
                data_name,
                folds,
                method,
                nf_mean,
                nf_std,
                acc_test_best,
                acc_test_mean,
                acc_test_std,
                acc_test_min,
                f1_test_mean,
                auc_test_mean,
                recall_test_mean,
                precision_test_mean,
                timecal_mean,
            ]
        )
        result_curve.append([data_name, method] + curve_mean.tolist()[0])
        result_run_best[:, method_idx] = acc_test[:, 0]
        results[data_name][method] = {
            "sf": sf,
            "nf": nf,
            "timecal": timecal,
            "acc_valid": acc_valid,
            "acc_test": acc_test,
            "f1_valid": f1_valid,
            "f1_test": f1_test,
            "auc_valid": auc_valid,
            "auc_test": auc_test,
            "valid_result": valid_result,
            "test_result": test_result,
            "curve_mean": curve_mean,
        }

        print(
            f"【{data_name}】【{method:<8}】NF: {nf_mean:.2f} ;   1-Fitness: {1 - fitness_mean:.2f} ;   "
            f"Acc_test: {acc_test_mean:.2f} ± {acc_test_std:.2f} ;   F1_test: {f1_test_mean * 100:.2f}% ;  "
            f"Auc_test:{auc_test_mean * 100:.2f}% ;  Time: {timecal_mean:.2f}"
        )

    if result_list:
        save_cv_results(
            data_name,
            folds,
            method_list,
            result_list,
            result_curve,
            result_run,
            result_run_best,
            result_run_best_name,
            current_date,
            output_dir,
            file_prefix,
        )

    return dataset_rows


def save_cv_results(
    data_name,
    folds,
    methodset,
    result_list,
    result_curve,
    result_run,
    result_run_best,
    result_run_best_name,
    current_date,
    output_dir="Result",
    file_prefix="cv",
):
    if not result_list:
        print(f"【{data_name}】无可保存的交叉验证结果。")
        return

    os.makedirs(output_dir, exist_ok=True)
    result_data = np.array(result_list, dtype=object)
    result_curve_arr = np.array(result_curve, dtype=object)
    result_best = np.concatenate((result_run_best_name, result_run_best), axis=1)
    result_run_data = np.array(
        [
            [
                item.get("Data Name"),
                item.get("Folds"),
                item.get("Method"),
                item.get("Run"),
                item.get("Feature Number"),
                item.get("Time Calculation"),
                item.get("Accuracy Valid"),
                item.get("Accuracy Test"),
                item.get("F1 Valid"),
                item.get("F1 Test"),
                item.get("AUC Valid"),
                item.get("AUC Test"),
            ]
            for item in result_run
        ],
        dtype=object,
    )

    result_data_header = [
        "Data Name",
        "Folds",
        "Method",
        "NF Mean",
        "NF Std",
        "Accuracy Test Best",
        "Accuracy Test Mean",
        "Accuracy Test Std",
        "Accuracy Test Min",
        "F1 Test Mean",
        "AUC Test Mean",
        "Recall Test Mean",
        "Precision Test Mean",
        "Timecal Mean",
    ]
    result_best_header = ["Data Name"] + methodset
    result_run_data_header = [
        "Data Name",
        "Folds",
        "Method",
        "Run",
        "Feature Number",
        "Time Calculation",
        "Accuracy Valid",
        "Accuracy Test",
        "F1 Valid",
        "F1 Test",
        "AUC Valid",
        "AUC Test",
    ]

    prefix = f"{file_prefix}_" if file_prefix else ""
    data_save_path = os.path.join(output_dir, f"{prefix}result_{current_date}.csv")
    curve_save_path = os.path.join(output_dir, f"{prefix}result_curve_{current_date}.csv")
    best_save_path = os.path.join(output_dir, f"{prefix}result_best_{current_date}.csv")
    run_save_path = os.path.join(output_dir, f"{prefix}result_run_{current_date}.csv")

    append_to_csv(data_save_path, result_data, delimiter=",", header=result_data_header)
    append_to_csv(curve_save_path, result_curve_arr, delimiter=",", header=None)
    append_to_csv(best_save_path, result_best, delimiter=",", header=result_best_header)
    append_to_csv(run_save_path, result_run_data, delimiter=",", header=result_run_data_header)

    print(f"【{data_name}】交叉验证结果已保存到 {output_dir} (文件前缀: {prefix or '无'})")


def save_results_to_csv(rows, output_dir="Result", file_name="cross_validation_results.csv"):
    if not rows:
        print("无可保存的交叉验证结果。")
        return
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    column_order = [
        "Dataset",
        "Samples",
        "Features",
        "Folds",
        "Method",
        "NF Mean",
        "NF Std",
        "Acc Test Best",
        "Acc Test Mean",
        "Acc Test Std",
        "Acc Test Min",
        "F1 Test Mean",
        "AUC Test Mean",
        "Recall Test Mean",
        "Precision Test Mean",
        "Time Mean",
        "Time Std",
        "Best Run",
        "Best Fold",
        "Best Valid Accuracy",
        "Best Valid Recall Macro",
        "Best Valid Precision Macro",
        "Best Valid F1 Macro",
        "Best Valid AUC Macro",
        "Best Test Accuracy",
        "Best Test Recall Macro",
        "Best Test Precision Macro",
        "Best Test F1 Macro",
        "Best Test AUC Macro",
        "Selected Feature Count",
        "Selected Feature Indices",
    ]
    df = df[[col for col in column_order if col in df.columns]]
    output_path = os.path.join(output_dir, file_name)
    df.to_csv(output_path, index=False)
    print(f"交叉验证结果已保存至 {output_path}")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DataConversionWarning)

    sys.path.append("Method-combination")

    dataset = [
        "Colon",
        "ALL_AML_4",
        "BreastGCE",
        "CML treatment",
        "BT1",
        "Leukemia2",
        "T9",
        "T11",
        "CNS",
        "LKM1",
        "Prostate",
        "LKM2",
    ]
    dataset = [filename.rstrip(".csv") for filename in dataset]
    # methodset = [ "PSO", "tmgwo", "essa", "SFE", "BBPSO", "VLPSO", "SFEPSO", "QLDGS-PSO", "rlpsoasm", "lapsodr", "igpso"]
    methodset = ["QLGSPSO"]
    desired_folds = 5
    runs = 2
    N = 20
    Maxiter = 5
    classify = "knn"
    cluster = ","
    split = 0
    func = 0
    opts = {
        "maxLt": runs,
        "N": N,
        "T": Maxiter,
        "cluster": cluster,
        "classify": classify,
        "split": split,
        "func": func,
        "knn_para": 3,
        "n_jobs_runs": 26,
        "n_jobs_folds": 1 # 每个 run 内折串行，避免双重并行占满
    }
    print("n_jobs_runs", opts["n_jobs_runs"])
    print("n_jobs_folds", opts["n_jobs_folds"])
    # current_date = datetime.datetime.now().strftime("%m-%d")
    current_date = "11"
    output_dir = "Result"
    file_prefix = "cv"
    for data_name in dataset:
        run_cross_validation(
            data_name,
            methodset,
            opts,
            desired_folds,
            current_date,
            runs=runs,
            output_dir=output_dir,
            file_prefix=file_prefix,
        )


if __name__ == "__main__":
    main()

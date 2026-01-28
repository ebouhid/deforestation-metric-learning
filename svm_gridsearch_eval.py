import os
import numpy as np
import pandas as pd
import datetime
import time
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from joblib import dump, Parallel, delayed, parallel_backend
from multiprocessing import Manager

seed = 42

def evaluate_svm(C, gamma, kernel, train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels, dataset):
    start_time = time.time()
    print(f"{datetime.datetime.now()} - Starting evaluation: dataset={dataset}, C={C}, gamma={gamma}, kernel={kernel}", flush=True)

    svm = SVC(C=C, gamma=gamma, kernel=kernel)
    svm.fit(train_embeddings, train_labels)
    train_predictions = svm.predict(train_embeddings)
    train_f1 = f1_score(train_labels, train_predictions)
    train_acc = accuracy_score(train_labels, train_predictions)
    train_bal_acc = balanced_accuracy_score(train_labels, train_predictions)
    train_precision = precision_score(train_labels, train_predictions)
    train_recall = recall_score(train_labels, train_predictions)
    train_tn, train_fp, train_fn, train_tp = confusion_matrix(train_labels, train_predictions).ravel()
    train_tnr = train_tn / (train_tn + train_fp) if (train_tn + train_fp) > 0 else 0

    val_predictions = svm.predict(val_embeddings)
    val_f1 = f1_score(val_labels, val_predictions)
    val_acc = accuracy_score(val_labels, val_predictions)
    val_bal_acc = balanced_accuracy_score(val_labels, val_predictions)
    val_precision = precision_score(val_labels, val_predictions)
    val_recall = recall_score(val_labels, val_predictions)
    val_tn, val_fp, val_fn, val_tp = confusion_matrix(val_labels, val_predictions).ravel()
    val_tnr = val_tn / (val_tn + val_fp) if (val_tn + val_fp) > 0 else 0

    test_predictions = svm.predict(test_embeddings)
    test_f1 = f1_score(test_labels, test_predictions)
    test_acc = accuracy_score(test_labels, test_predictions)
    test_bal_acc = balanced_accuracy_score(test_labels, test_predictions)
    test_precision = precision_score(test_labels, test_predictions)
    test_recall = recall_score(test_labels, test_predictions)
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(test_labels, test_predictions).ravel()
    test_tnr = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0

    elapsed_time = time.time() - start_time
    print(f"{datetime.datetime.now()} - Finished: dataset={dataset}, C={C}, gamma={gamma}, kernel={kernel} "
          f"in {elapsed_time:.2f} seconds. Test F1: {test_f1:.4f}", flush=True)

    model_filename = f"svm_C{C}_gamma{gamma}_kernel{kernel}.joblib"
    model_path = os.path.join(f"svm_models/{dataset}", model_filename)
    dump(svm, model_path)

    return {
        'dataset': dataset,
        'C': C,
        'gamma': gamma,
        'kernel': kernel,
        'train_f1': train_f1,
        'train_acc': train_acc,
        'train_bal_acc': train_bal_acc,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_tnr': train_tnr,
        'val_f1': val_f1,
        'val_acc': val_acc,
        'val_bal_acc': val_bal_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_tnr': val_tnr,
        'test_f1': test_f1,
        'test_acc': test_acc,
        'test_bal_acc': test_bal_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_tnr': test_tnr,
        'elapsed_time': elapsed_time,
        'model_path': model_path,
        'is_best': False
    }

def append_results_to_csv(results, csv_path, lock):
    with lock:  # Ensure thread-safe writing
        pd.DataFrame([results]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

if __name__ == "__main__":
    embeddings_to_evaluate = ['r18dml654', 'r50dml654', 'har654', 'r101dml654']
    # embeddings_to_evaluate = ['har654']
    savepath = "svm_gridsearch_results.csv"

    # Initialize multiprocessing manager
    manager = Manager()
    lock = manager.Lock()

    for dataset in embeddings_to_evaluate:
        print(f"Evaluating dataset: {dataset}")
        embeddings_path = f"embeddings_{dataset}/"

        def load_embeddings(folder):
            embeddings, labels = [], []
            for emb in os.listdir(os.path.join(embeddings_path, folder, "forest")):
                embeddings.append(np.load(os.path.join(embeddings_path, folder, "forest", emb)).flatten())
                labels.append(0)
            for emb in os.listdir(os.path.join(embeddings_path, folder, "recent_def")):
                embeddings.append(np.load(os.path.join(embeddings_path, folder, "recent_def", emb)).flatten())
                labels.append(1)
            return np.array(embeddings), np.array(labels)

        train_embeddings, train_labels = load_embeddings("train")
        val_embeddings, val_labels = load_embeddings("val")
        test_embeddings, test_labels = load_embeddings("test")

        train_embeddings = normalize(train_embeddings)
        val_embeddings = normalize(val_embeddings)
        test_embeddings = normalize(test_embeddings)

        os.makedirs(f"svm_models/{dataset}", exist_ok=True)

        param_grid = {
            'C': [0.05, 0.1, 1, 10, 100, 1000],
            'gamma': [0.1, 1, 'auto'],
            'kernel': ['rbf', 'linear', 'poly']
        }

        # Initialize CSV with headers
        if not os.path.exists(savepath):
            pd.DataFrame(columns=[
                'dataset', 'C', 'gamma', 'kernel', 'train_f1', 'train_acc', 'train_bal_acc', 'train_precision', 'train_recall', 'train_tnr',
                'val_f1', 'val_acc', 'val_bal_acc', 'val_precision', 'val_recall', 'val_tnr',
                'test_f1', 'test_acc', 'test_bal_acc', 'test_precision', 'test_recall', 'test_tnr',
                'elapsed_time', 'model_path', 'is_best'
            ]).to_csv(savepath, index=False)

        with parallel_backend('loky', n_jobs=36):
            jobs = Parallel(verbose=10)(
                delayed(lambda c, g, k: append_results_to_csv(
                    evaluate_svm(c, g, k, train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels, dataset),
                    savepath,
                    lock
                ))(C, gamma, kernel)
                for C in param_grid['C']
                for gamma in param_grid['gamma']
                for kernel in param_grid['kernel']
            )

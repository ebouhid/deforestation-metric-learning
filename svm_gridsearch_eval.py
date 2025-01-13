import os
import numpy as np
import pandas as pd
import datetime
import time
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from joblib import dump, Parallel, delayed, parallel_backend
from multiprocessing import Manager

seed = 42

def evaluate_svm(C, gamma, kernel, train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels, dataset):
    start_time = time.time()
    print(f"{datetime.datetime.now()} - Starting evaluation: dataset={dataset}, C={C}, gamma={gamma}, kernel={kernel}", flush=True)

    svm = SVC(C=C, gamma=gamma, kernel=kernel)
    svm.fit(train_embeddings, train_labels)
    val_predictions = svm.predict(val_embeddings)
    val_f1 = f1_score(val_labels, val_predictions)
    val_acc = accuracy_score(val_labels, val_predictions)
    val_bal_acc = balanced_accuracy_score(val_labels, val_predictions)
    val_precision = precision_score(val_labels, val_predictions)
    val_recall = recall_score(val_labels, val_predictions)

    test_predictions = svm.predict(test_embeddings)
    test_f1 = f1_score(test_labels, test_predictions)
    test_acc = accuracy_score(test_labels, test_predictions)
    test_bal_acc = balanced_accuracy_score(test_labels, test_predictions)
    test_precision = precision_score(test_labels, test_predictions)
    test_recall = recall_score(test_labels, test_predictions)

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
        'val_f1': val_f1,
        'val_acc': val_acc,
        'val_bal_acc': val_bal_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'test_f1': test_f1,
        'test_acc': test_acc,
        'test_bal_acc': test_bal_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'elapsed_time': elapsed_time,
        'model_path': model_path,
        'is_best': False
    }

def append_results_to_csv(results, csv_path, lock):
    with lock:  # Ensure thread-safe writing
        pd.DataFrame([results]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

if __name__ == "__main__":
    embeddings_to_evaluate = ['r18dml', 'r50dml', 'har', 'r101dml']
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
            'C': [0.1, 1, 10],
            'gamma': [0.1, 1, 'auto'],
            'kernel': ['rbf', 'linear', 'poly']
        }

        # Initialize CSV with headers
        if not os.path.exists(savepath):
            pd.DataFrame(columns=[
                'dataset', 'C', 'gamma', 'kernel', 'val_f1', 'val_acc', 'val_bal_acc', 'val_precision',
                'val_recall', 'test_f1', 'test_acc', 'test_bal_acc', 'test_precision', 'test_recall',
                'elapsed_time', 'model_path', 'is_best'
            ]).to_csv(savepath, index=False)

        with parallel_backend('loky', n_jobs=32):
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

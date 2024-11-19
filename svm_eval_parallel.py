import os
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from joblib import dump
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import datetime

seed = 42

def train_and_evaluate_svm(dataset):
    # Load embeddings
    embeddings_path = f"embeddings_{dataset}/"
    
    # Load train embeddings and labels
    train_embeddings = []
    train_labels = []
    for forest_emb in os.listdir(os.path.join(embeddings_path, "train", "forest")):
        emb = np.load(os.path.join(embeddings_path, "train", "forest", forest_emb))
        train_embeddings.append(emb.flatten())
        train_labels.append(0)
    for recent_def_emb in os.listdir(os.path.join(embeddings_path, "train", "recent_def")):
        emb = np.load(os.path.join(embeddings_path, "train", "recent_def", recent_def_emb))
        train_embeddings.append(emb.flatten())
        train_labels.append(1)
    
    # Load validation embeddings and labels
    val_embeddings = []
    val_labels = []
    for forest_emb in os.listdir(os.path.join(embeddings_path, "val", "forest")):
        emb = np.load(os.path.join(embeddings_path, "val", "forest", forest_emb))
        val_embeddings.append(emb.flatten())
        val_labels.append(0)
    for recent_def_emb in os.listdir(os.path.join(embeddings_path, "val", "recent_def")):
        emb = np.load(os.path.join(embeddings_path, "val", "recent_def", recent_def_emb))
        val_embeddings.append(emb.flatten())
        val_labels.append(1)

    # Convert lists to numpy arrays
    train_embeddings = np.array(train_embeddings)
    train_labels = np.array(train_labels)
    val_embeddings = np.array(val_embeddings)
    val_labels = np.array(val_labels)

    # Normalize embeddings
    train_embeddings = normalize(train_embeddings)
    val_embeddings = normalize(val_embeddings)
    
    # Define and train SVM
    best_svm = SVC(
        C=100,
        kernel='rbf',
        class_weight='balanced',
        random_state=seed,
        verbose=False
    )
    print(f"{datetime.datetime.now()} | Training SVM on dataset {dataset}")
    train_start = datetime.datetime.now()
    best_svm.fit(train_embeddings, train_labels)
    print(f"{datetime.datetime.now()} | Training completed in {datetime.datetime.now() - train_start}s")

    # Validate SVM
    print(f"{datetime.datetime.now()} | Validating SVM on dataset {dataset}")
    val_start = datetime.datetime.now()
    val_predictions = best_svm.predict(val_embeddings)
    print(f"{datetime.datetime.now()} | Validation completed in {datetime.datetime.now() - val_start}s")
    
    # Calculate metrics
    val_f1 = f1_score(val_labels, val_predictions)
    val_acc = accuracy_score(val_labels, val_predictions)
    val_bal_acc = balanced_accuracy_score(val_labels, val_predictions)
    val_precision = precision_score(val_labels, val_predictions)
    val_recall = recall_score(val_labels, val_predictions)

    # Save model
    os.makedirs("svm_models", exist_ok=True)
    model_path = f"svm_models/svm_{dataset}.joblib"
    dump(best_svm, model_path)
    print(f"Model saved to {model_path}")

    # Return results
    return {
        'dataset': dataset,
        'C': 100,
        'gamma': 'auto',
        'val_f1': val_f1,
        'val_acc': val_acc,
        'val_bal_acc': val_bal_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'model_path': model_path
    }

if __name__ == "__main__":
    embeddings_to_evaluate = ['har', 'r18', 'r50', 'r18ft', 'r50ft']
    results = []

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor() as executor:
        future_to_dataset = {executor.submit(train_and_evaluate_svm, dataset): dataset for dataset in embeddings_to_evaluate}
        
        for future in as_completed(future_to_dataset):
            dataset = future_to_dataset[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing dataset {dataset}: {e}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("svm_grid_search_results_normalized.csv", index=False)
    print("Results saved to svm_grid_search_results_normalized.csv")

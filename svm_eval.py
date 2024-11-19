import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import datetime
from joblib import dump

seed = 42

if __name__ == "__main__":
    embeddings_to_evaluate = ['har', 'r18', 'r50', 'r18ft', 'r50ft']
    results = []

    for dataset in embeddings_to_evaluate:
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
        

        # # Define SVM and hyperparameter grid
        # svm = SVC()
        # param_grid = {
        #     'C': [0.1, 1, 10],
        #     'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        #     'kernel': ['rbf', 'linear', 'poly']
        # }

        # # Set up grid search with cross-validation using F1 score
        # grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1', n_jobs=-1)
        # grid_search.fit(train_embeddings, train_labels)
        
        # # Get best estimator and validate
        # best_svm = grid_search.best_estimator_
        best_svm = SVC(C=100,
              kernel='rbf',
              class_weight='balanced',
              random_state=seed,
              verbose=True
        )
        print(f"\n{datetime.datetime.now()} | Training SVM on dataset {dataset}")
        train_start = datetime.datetime.now()
        best_svm.fit(train_embeddings, train_labels)
        print(f"{datetime.datetime.now()} | Training completed in {datetime.datetime.now() - train_start}s")
        print(f"{datetime.datetime.now()} | Validating SVM on dataset {dataset}")
        val_start = datetime.datetime.now()
        val_predictions = best_svm.predict(val_embeddings)
        print(f"{datetime.datetime.now()} | Validation completed in {datetime.datetime.now() - val_start}s")
        
        val_f1 = f1_score(val_labels, val_predictions)
        val_acc = accuracy_score(val_labels, val_predictions)
        val_bal_acc = balanced_accuracy_score(val_labels, val_predictions)
        val_precision = precision_score(val_labels, val_predictions)
        val_recall = recall_score(val_labels, val_predictions)


        
        # # Store results in a dictionary
        # for params, mean_test_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        #     result = {
        #         'dataset': dataset,
        #         'C': params['C'],
        #         'gamma': params['gamma'],
        #         'mean_cv_f1': mean_test_score,
        #         'val_f1': val_f1 if params == grid_search.best_params_ else None
        #     }
        #     results.append(result)

        # Store results in a dictionary
        result = {
            'dataset': dataset,
            'C': 100,
            'gamma': 'auto',
            # 'mean_cv_f1': None,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'val_bal_acc': val_bal_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'model_path': f"svm_models/svm_{dataset}.joblib"
        }
        results.append(result)

        # Save trained model
        os.makedirs("svm_models", exist_ok=True)
        dump(best_svm, f"svm_models/svm_{dataset}.joblib")
        print(f"Model saved to svm_models/svm_{dataset}.joblib")

    # Convert results to a DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("svm_grid_search_results_normalized.csv", index=False)
    print("Results saved to svm_grid_search_results_normalized.csv")

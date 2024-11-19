import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
df = pd.read_csv("./model_metrics_norsz.csv")

# List of metrics to plot
metrics = ["f1", "recall", "precision", "balanced_acc"]

# Group by model to handle multiple models in the same file
for model_name, model_data in df.groupby("model"):
    # Create a directory for each model if it doesn't exist
    os.makedirs(f"{model_name}_training_results", exist_ok=True)
    
    # Separate train and val data
    train_data = model_data[model_data["loop"] == "train"]
    val_data = model_data[model_data["loop"] == "val"]
    
    # Plot each metric
    for metric in metrics:
        plt.figure()
        plt.plot(train_data["epoch"], train_data[metric], label="Train", marker='o')
        plt.plot(val_data["epoch"], val_data[metric], label="Validation", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(f"{model_name} - {metric.capitalize()} over Epochs")
        plt.legend()
        
        # Save the plot
        plt.savefig(f"{model_name}_training_results/{metric}_curve.png")
        plt.close()

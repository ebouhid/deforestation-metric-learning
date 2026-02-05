import torch
import timm
from dataset import SegmentClassificationDataset
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from umap_to_video import images_to_video


def validate(model, train_loader, val_loader, test_loader, loss_func, device, epoch, result_dir, backbone_name, miner_type):
    """
    Validates the model on the validation set (computing the mean loss) and also collects embeddings
    from train, validation, and test sets to create a combined UMAP visualization. 
    Composite labels are used to distinguish each set and class combination.
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    # Containers for embeddings and composite labels for the combined UMAP plot
    combined_embeddings = []
    combined_labels = []  # Composite labels (e.g., "train_0", "val_1", "test_0")

    # First, process the validation set to compute loss and collect embeddings
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating (val set)"):
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device)

            embeddings = model(data)
            loss = loss_func(embeddings, labels)
            total_loss += loss.item()
            num_batches += 1

            # Append embeddings and composite labels for validation set
            emb_np = embeddings.cpu().numpy()
            lab_np = labels.cpu().numpy()
            combined_embeddings.append(emb_np)
            combined_labels.extend([f"val_{int(lbl)}" for lbl in lab_np])

    mean_loss = total_loss / num_batches

    # Helper function to collect embeddings from a given loader
    def collect_embeddings(loader, set_name):
        embeddings_list = []
        composite_labels = []
        with torch.no_grad():
            for data, labels in tqdm(loader, desc=f"Collecting embeddings ({set_name} set)"):
                data = data.to(device)
                labels = labels.to(device)
                embeddings = model(data)
                embeddings_list.append(embeddings.cpu().numpy())
                composite_labels.extend([f"{set_name}_{int(lbl)}" for lbl in labels.cpu().numpy()])
        if embeddings_list:
            return np.concatenate(embeddings_list, axis=0), composite_labels
        else:
            return np.array([]), []

    # Collect embeddings from the train and test sets
    train_emb, train_labels = collect_embeddings(train_loader, "train")
    test_emb, test_labels = collect_embeddings(test_loader, "test")

    # Append train and test embeddings to the combined list
    if train_emb.size:
        combined_embeddings.append(train_emb)
        combined_labels.extend(train_labels)
    if test_emb.size:
        combined_embeddings.append(test_emb)
        combined_labels.extend(test_labels)

    # Concatenate all embeddings and composite labels
    all_embeddings = np.concatenate(combined_embeddings, axis=0)
    all_labels = np.array(combined_labels)

    # Apply UMAP for dimensionality reduction
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(all_embeddings)

    # Define color mapping for six composite groups
    color_mapping = {
        "train_0": "#1f77b4",  # Train Forest
        "train_1": "#ff7f0e",  # Train Nonforest
        "val_0":   "#2ca02c",  # Val Forest
        "val_1":   "#d62728",  # Val Nonforest
        "test_0":  "#9467bd",  # Test Forest
        "test_1":  "#8c564b",  # Test Nonforest
    }
    # Map composite labels to colors
    point_colors = [color_mapping[label] for label in all_labels]

    # Create combined UMAP scatter plot
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=point_colors, s=50)
    plt.title(f"UMAP of Embeddings (Train, Val, Test) - Epoch {epoch}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # Create custom legend using matplotlib patches
    from matplotlib.patches import Patch
    legend_elements = []
    for comp_label, color in color_mapping.items():
        dataset, label = comp_label.split("_")
        label_str = "Forest" if label == "0" else "Nonforest"
        legend_elements.append(Patch(facecolor=color, edgecolor=color, label=f"{dataset.capitalize()} {label_str}"))
    plt.legend(handles=legend_elements, loc="best", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    # Save the combined UMAP plot
    umap_save_path = os.path.join(
        result_dir, backbone_name, f"umap_embeddings_{miner_type}", f"umap_all_embeddings_{miner_type}_ep{epoch}.png"
    )
    os.makedirs(os.path.dirname(umap_save_path), exist_ok=True)
    plt.savefig(umap_save_path)
    plt.close()

    return mean_loss


def validate_old(model, val_loader, loss_func, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            data, labels = batch
            embeddings = model(data.to(device))
            loss = loss_func(embeddings, labels)
            total_loss += loss.item()
            num_batches += 1

    mean_loss = total_loss / num_batches
    return mean_loss


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    avg_loss = 0.0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )
        avg_loss += loss.item() / len(train_loader)
    
    return avg_loss


if __name__ == "__main__":
    result_dir = "dml_results_sentinelGECCO-256sq-run0"
    for backbone_name in ["resnet18", "resnet50", "resnet101"]:
        os.makedirs(os.path.join(result_dir, backbone_name), exist_ok=True)
        os.makedirs(os.path.join(result_dir, backbone_name, "models"), exist_ok=True)
        for miner_type in ["all"]:
            # Set up dataset
            root_dir = "dml_dataset_sentinelGECCO-256sq/"
            train_regions = ["x01", "x02", "x06", "x08", "x10"]
            val_regions = ["x07", "x09"]
            test_regions = ["x03", "x04"]
            transform = True
            save_every_epoch = True

            train_dataset = SegmentClassificationDataset(
                root_dir=root_dir, regions=train_regions, transform=transform, return_dict=False, exc_labels=[8])
            val_dataset = SegmentClassificationDataset(
                root_dir=root_dir, regions=val_regions, transform=False, return_dict=False, exc_labels=[8])
            test_dataset = SegmentClassificationDataset(
                root_dir=root_dir, regions=test_regions, transform=False, return_dict=False, exc_labels=[8])

            print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
            # DataLoader with Sampler (only for training)
            batch_size = 64
            train_sampler = MPerClassSampler(labels=train_dataset.labels, m=8, length_before_new_iter=len(train_dataset))
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler=train_sampler)
            # Val/Test loaders without sampler for complete, deterministic evaluation
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

            # Set up embedding model
            class TrunkModel(torch.nn.Module):
                def __init__(self, backbone_name="resnet50", pretrained=True):
                    super(TrunkModel, self).__init__()
                    self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

                def forward(self, x):
                    return self.backbone(x)
            
            class EmbedderModel(torch.nn.Module):
                def __init__(self, input_size, embedding_size):
                    super(EmbedderModel, self).__init__()
                    self.embedding_layer = torch.nn.Linear(input_size, embedding_size)

                def forward(self, x):
                    return self.embedding_layer(x)
            
            embedding_size = timm.create_model(backbone_name, pretrained=True).fc.in_features
            trunk = TrunkModel(backbone_name=backbone_name)
            embedder = EmbedderModel(input_size=trunk.backbone.num_features, embedding_size=embedding_size)
            device = torch.device("cuda:0")
            model = torch.nn.Sequential(trunk, embedder).to(device)

            # Loss and Miner
            loss = TripletMarginLoss(margin=0.2)
            miner = TripletMarginMiner(margin=0.2, type_of_triplets=miner_type)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            
            initial_val_loss = validate(model, train_loader, val_loader, test_loader, loss, device, 0, result_dir, backbone_name, miner_type)
            print(f"Initial Validation Mean Loss: {initial_val_loss:.4f}")

            num_epochs = 200
            val_data = []
            train_data = []
            best_val_loss = float("inf")
            for epoch in range(1, num_epochs + 1):
                print(f"Epoch {epoch}/{num_epochs} | Miner: {miner.type_of_triplets}")
                train_mean_loss = train(model, loss, miner, device, train_loader, optimizer, epoch)
                train_data.append({"epoch": epoch, "loss": train_mean_loss})
                val_loss = validate(model, train_loader, val_loader, test_loader, loss, device, epoch, result_dir, backbone_name, miner_type)
                val_data.append({"epoch": epoch, "val_loss": val_loss})
                print(f"Validation Mean Loss on epoch {epoch}: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(result_dir, backbone_name, "models", f"best_model_state_dict_{miner_type}_epoch-{epoch}.pth"))
                    print("New best model saved!")
                if save_every_epoch:
                    torch.save(model.state_dict(), os.path.join(result_dir, backbone_name, "models", f"model_state_dict_{miner_type}_epoch-{epoch}.pth"))
            print("Training complete")

            final_val_loss = validate(model, train_loader, val_loader, test_loader, loss, device, epoch, result_dir, backbone_name, miner_type)
            print(f"Final Validation Mean Loss: {final_val_loss:.4f}")

            training_info = pd.DataFrame(train_data)
            val_info = pd.DataFrame(val_data)
            plt.figure(figsize=(12, 6))
            plt.plot(training_info["epoch"], training_info["loss"], label="Training Loss")
            plt.savefig(os.path.join(result_dir, backbone_name, f"training_loss{miner_type}.png"))
            plt.clf()
            plt.figure(figsize=(12, 6))
            plt.plot(val_info["epoch"], val_info["val_loss"], label="Validation Loss")
            plt.savefig(os.path.join(result_dir, backbone_name, f"validation_loss{miner_type}.png"))

            training_info.to_csv(os.path.join(result_dir, backbone_name, f"training_info_{miner_type}.csv"), index=False)
            val_info.to_csv(os.path.join(result_dir, backbone_name, f"val_info_{miner_type}.csv"), index=False)

            images_to_video(
                image_folder=os.path.join(result_dir, backbone_name, f"umap_embeddings_{miner_type}"),
                output_video=os.path.join(result_dir, backbone_name, f"umap_embeddings_{miner_type}.mp4"),
                fps=15,
                resolution=None
            )

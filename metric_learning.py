import torch
import timm
from dataset import SegmentClassificationDataset, CLASS_NAMES
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.samplers import MPerClassSampler
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from umap_to_video import images_to_video


def compute_nn_metrics(embeddings, labels):
    distances = torch.cdist(embeddings, embeddings)
    distances.fill_diagonal_(float("inf"))
    nearest_indices = distances.argmin(dim=1)
    nearest_labels = labels[nearest_indices]
    correct = nearest_labels == labels

    prec_at_1 = correct.float().mean().item() if labels.numel() > 0 else 0.0
    recalls = []
    for cls in labels.unique():
        mask = labels == cls
        if mask.any():
            recalls.append(correct[mask].float().mean())
    balanced_acc = torch.stack(recalls).mean().item() if recalls else 0.0

    return prec_at_1, balanced_acc


def validate(model, train_loader, val_loader, test_loader, loss_func, mining_func, device, epoch, result_dir, backbone_name, miner_type):
    """
    Validates the model on the validation set (computing the mean loss) and also collects embeddings
    from train, validation, and test sets to create a combined UMAP visualization. 
    Composite labels are used to distinguish each set and class combination.
    """
    model.eval()
    def evaluate_loader(loader, set_name):
        total_loss = 0.0
        num_batches = 0
        embeddings_list = []
        labels_list = []
        embeddings_np = []
        composite_labels = []
        with torch.no_grad():
            for data, labels in tqdm(loader, desc=f"Evaluating ({set_name} set)"):
                data = data.to(device)
                labels = labels.to(device)

                embeddings = model(data)
                indices_tuple = mining_func(embeddings, labels)
                loss = loss_func(embeddings, labels, indices_tuple)
                total_loss += loss.item()
                num_batches += 1

                embeddings_list.append(embeddings)
                labels_list.append(labels)

                emb_np = embeddings.cpu().numpy()
                embeddings_np.append(emb_np)
                composite_labels.extend([f"{set_name}_{int(lbl)}" for lbl in labels.cpu().numpy()])

        mean_loss = total_loss / max(1, num_batches)
        if embeddings_list:
            all_emb = torch.cat(embeddings_list)
            all_lab = torch.cat(labels_list)
            prec_at_1, balanced_acc = compute_nn_metrics(all_emb, all_lab)
        else:
            prec_at_1, balanced_acc = 0.0, 0.0
        if embeddings_np:
            embeddings_np = np.concatenate(embeddings_np, axis=0)
        else:
            embeddings_np = np.array([])

        return mean_loss, prec_at_1, balanced_acc, embeddings_np, composite_labels

    val_loss, val_prec_at_1, val_bal_acc, val_emb, val_labels = evaluate_loader(val_loader, "val")
    test_loss, test_prec_at_1, test_bal_acc, test_emb, test_labels = evaluate_loader(test_loader, "test")

    print(
        f"Validation Loss: {val_loss:.4f} | Precision@1: {val_prec_at_1:.4f} | Balanced Acc: {val_bal_acc:.4f}"
    )
    print(
        f"Test Loss: {test_loss:.4f} | Precision@1: {test_prec_at_1:.4f} | Balanced Acc: {test_bal_acc:.4f}"
    )

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

    # Collect embeddings from train set for UMAP
    train_emb, train_composite = collect_embeddings(train_loader, "train")
    
    # Combine all embeddings
    all_embeddings = np.concatenate([train_emb, val_emb, test_emb], axis=0)
    all_composite_labels = train_composite + val_labels + test_labels
    
    # Create UMAP embedding
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    umap_embeddings = reducer.fit_transform(all_embeddings)
    
    # Create output directories for per-set UMAP plots
    umap_root_dir = os.path.join(result_dir, backbone_name, f"umap_embeddings_{miner_type}")
    umap_dirs = {
        "train": os.path.join(umap_root_dir, "train"),
        "val": os.path.join(umap_root_dir, "val"),
        "test": os.path.join(umap_root_dir, "test"),
    }
    for path in umap_dirs.values():
        os.makedirs(path, exist_ok=True)

    # Parse composite labels to get set type and class
    set_types = [label.split('_')[0] for label in all_composite_labels]
    class_ids = [int(label.split('_')[1]) for label in all_composite_labels]

    # Fixed color scheme based on CLASS_NAMES ordering
    fixed_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    ordered_class_ids = sorted(CLASS_NAMES.keys())
    class_to_color = {
        class_id: fixed_colors[i % len(fixed_colors)]
        for i, class_id in enumerate(ordered_class_ids)
    }

    def plot_umap_for_set(set_name, marker, alpha):
        plt.figure(figsize=(12, 10))
        unique_classes = sorted(set(class_ids))
        for cls in unique_classes:
            mask = [(s == set_name and c == cls) for s, c in zip(set_types, class_ids)]
            if any(mask):
                indices = [i for i, m in enumerate(mask) if m]
                plt.scatter(
                    umap_embeddings[indices, 0],
                    umap_embeddings[indices, 1],
                    c=[class_to_color.get(cls, "#000000")],
                    marker=marker,
                    alpha=alpha,
                    s=30,
                    label=CLASS_NAMES.get(cls, f"Class {cls}")
                )

        plt.title(f"UMAP Embeddings - {set_name.capitalize()} - Epoch {epoch}")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        output_path = os.path.join(umap_dirs[set_name], f"epoch_{epoch}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"UMAP visualization saved to {output_path}")

    plot_umap_for_set("train", marker="o", alpha=0.4)
    plot_umap_for_set("val", marker="s", alpha=0.7)
    plot_umap_for_set("test", marker="^", alpha=0.7)

    return val_loss, val_prec_at_1, val_bal_acc, test_loss, test_prec_at_1, test_bal_acc


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
    for backbone_name in ["resnet101"]:#["resnet18", "resnet50", "resnet101"]:
        os.makedirs(os.path.join(result_dir, backbone_name), exist_ok=True)
        os.makedirs(os.path.join(result_dir, backbone_name, "models"), exist_ok=True)
        for miner_type in ["all"]:
            # Set up dataset
            root_dir = "dml_dataset_sentinelGECCO-256sq/"
            train_regions = ["x02", "x06", "x08", "x10"]
            val_regions = ["x01", "x07", "x09"]
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
            
            (
                initial_val_loss,
                initial_val_acc,
                initial_val_bal_acc,
                initial_test_loss,
                initial_test_acc,
                initial_test_bal_acc,
            ) = validate(
                model,
                train_loader,
                val_loader,
                test_loader,
                loss,
                miner,
                device,
                0,
                result_dir,
                backbone_name,
                miner_type,
            )
            print(
                f"Initial Val Loss: {initial_val_loss:.4f} | Val Balanced Acc: {initial_val_bal_acc:.4f} | "
                f"Test Loss: {initial_test_loss:.4f} | Test Balanced Acc: {initial_test_bal_acc:.4f}"
            )

            num_epochs = 10
            val_data = []
            train_data = []
            test_data = []
            best_val_loss = float("inf")
            for epoch in range(1, num_epochs + 1):
                print(f"Epoch {epoch}/{num_epochs} | Miner: {miner.type_of_triplets} | LR: {scheduler.get_last_lr()[0]:.6f}")
                train_mean_loss = train(model, loss, miner, device, train_loader, optimizer, epoch)
                train_data.append({"epoch": epoch, "loss": train_mean_loss})
                (
                    val_loss,
                    val_acc,
                    val_bal_acc,
                    test_loss,
                    test_acc,
                    test_bal_acc,
                ) = validate(
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    loss,
                    miner,
                    device,
                    epoch,
                    result_dir,
                    backbone_name,
                    miner_type,
                )
                val_data.append(
                    {
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_bal_acc": val_bal_acc,
                    }
                )
                test_data.append(
                    {
                        "epoch": epoch,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "test_bal_acc": test_bal_acc,
                    }
                )
                print(
                    f"Validation Mean Loss on epoch {epoch}: {val_loss:.4f} | "
                    f"Balanced Acc: {val_bal_acc:.4f} | Test Balanced Acc: {test_bal_acc:.4f}"
                )
                scheduler.step()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(result_dir, backbone_name, "models", f"best_model_state_dict_{miner_type}_epoch-{epoch}.pth"))
                    print("New best model saved!")
                if save_every_epoch:
                    torch.save(model.state_dict(), os.path.join(result_dir, backbone_name, "models", f"model_state_dict_{miner_type}_epoch-{epoch}.pth"))
            print("Training complete")

            (
                final_val_loss,
                final_val_acc,
                final_val_bal_acc,
                final_test_loss,
                final_test_acc,
                final_test_bal_acc,
            ) = validate(
                model,
                train_loader,
                val_loader,
                test_loader,
                loss,
                miner,
                device,
                epoch,
                result_dir,
                backbone_name,
                miner_type,
            )
            print(
                f"Final Val Loss: {final_val_loss:.4f} | Val Balanced Acc: {final_val_bal_acc:.4f} | "
                f"Final Test Balanced Acc: {final_test_bal_acc:.4f}"
            )

            training_info = pd.DataFrame(train_data)
            val_info = pd.DataFrame(val_data)
            test_info = pd.DataFrame(test_data)
            plt.figure(figsize=(12, 6))
            plt.plot(training_info["epoch"], training_info["loss"], label="Training Loss")
            plt.savefig(os.path.join(result_dir, backbone_name, f"training_loss{miner_type}.png"))
            plt.clf()
            plt.figure(figsize=(12, 6))
            plt.plot(val_info["epoch"], val_info["val_loss"], label="Validation Loss")
            plt.savefig(os.path.join(result_dir, backbone_name, f"validation_loss{miner_type}.png"))
            plt.clf()
            plt.figure(figsize=(12, 6))
            plt.plot(val_info["epoch"], val_info["val_bal_acc"], label="Validation Balanced Acc")
            plt.plot(test_info["epoch"], test_info["test_bal_acc"], label="Test Balanced Acc")
            plt.savefig(os.path.join(result_dir, backbone_name, f"balanced_accuracy{miner_type}.png"))

            training_info.to_csv(os.path.join(result_dir, backbone_name, f"training_info_{miner_type}.csv"), index=False)
            val_info.to_csv(os.path.join(result_dir, backbone_name, f"val_info_{miner_type}.csv"), index=False)
            test_info.to_csv(os.path.join(result_dir, backbone_name, f"test_info_{miner_type}.csv"), index=False)

            images_to_video(
                image_folder=os.path.join(result_dir, backbone_name, f"umap_embeddings_{miner_type}", "train"),
                output_video=os.path.join(result_dir, backbone_name, f"umap_embeddings_{miner_type}_train.mp4"),
                fps=15,
                resolution=None
            )
            images_to_video(
                image_folder=os.path.join(result_dir, backbone_name, f"umap_embeddings_{miner_type}", "val"),
                output_video=os.path.join(result_dir, backbone_name, f"umap_embeddings_{miner_type}_val.mp4"),
                fps=15,
                resolution=None
            )
            images_to_video(
                image_folder=os.path.join(result_dir, backbone_name, f"umap_embeddings_{miner_type}", "test"),
                output_video=os.path.join(result_dir, backbone_name, f"umap_embeddings_{miner_type}_test.mp4"),
                fps=15,
                resolution=None
            )

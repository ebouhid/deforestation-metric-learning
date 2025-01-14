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


def validate(model, val_loader, loss_func, device):
        model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Load data
                data, labels = batch

                # Compute embeddings
                embeddings = model(data.to(device))

                # Compute loss
                loss = loss_func(embeddings, labels)
                total_loss += loss.item()
                num_batches += 1

        mean_loss = total_loss / num_batches
        
        return mean_loss

def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    avg_loss = 0.0
    # print(f"Length of train_loader: {len(train_loader)}")
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
    for backbone_name in ["resnet18", "resnet50", "resnet101"]:
        os.makedirs(os.path.join("dml_results", backbone_name), exist_ok=True)
        os.makedirs(os.path.join("dml_results", backbone_name, "models"), exist_ok=True)
        for miner_type in ["semihard", "hard", "all"]:
            # Set up dataset
            root_dir = "./segment_embeddings_classification_dataset_norsz/"
            train_regions = ["x02", "x06", "x07", "x09", "x10"]
            val_regions = ["x03", "x08"]
            test_regions = ["x01", "x04"]
            transform = True
            save_every_epoch = True

            train_dataset = SegmentClassificationDataset(
                root_dir=root_dir, regions=train_regions, transform=transform, return_dict=False, exc_labels=[-1])
            val_dataset = SegmentClassificationDataset(
                root_dir=root_dir, regions=val_regions, transform=False, return_dict=False, exc_labels=[-1])
            
            print(f"Len Train Dataset: {len(train_dataset)} | Len Val Dataset: {len(val_dataset)}")
            print(50 * '=')
            print(f"Forest samples in train dataset: {sum([1 for x in train_dataset if x[1] == 0])}")
            print(f"Recent deforestation samples in train dataset: {sum([1 for x in train_dataset if x[1] == 1])}")
            print(50 * "-")
            print(f"Forest samples in val dataset: {sum([1 for x in val_dataset if x[1] == 0])}")
            print(f"Recent deforestation samples in val dataset: {sum([1 for x in val_dataset if x[1] == 1])}")

            # DataLoader with Sampler
            batch_size = 256

            train_sampler = MPerClassSampler(labels=train_dataset.labels, m=batch_size//2, length_before_new_iter=len(train_dataset))
            val_sampler = MPerClassSampler(labels=val_dataset.labels, m=batch_size//2, length_before_new_iter=len(val_dataset))
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, num_workers=32, sampler=train_sampler)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False, sampler=val_sampler)

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
            
            # Define embedding size based off backbone
            embedding_size = timm.create_model(backbone_name, pretrained=True).fc.in_features
            # backbone_name = "resnet50"
            trunk = TrunkModel(backbone_name=backbone_name)
            embedder = EmbedderModel(input_size=trunk.backbone.num_features, embedding_size=embedding_size)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model = torch.nn.Sequential(trunk, embedder).to(device)

            # Loss and Miner
            loss = TripletMarginLoss(margin=0.2)
            miner = TripletMarginMiner(margin=0.2, type_of_triplets=miner_type)

            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            
            initial_val_loss = validate(model, val_loader, loss, device)
            print(f"Initial Validation Mean Loss: {initial_val_loss:.4f}")

            # Train
            num_epochs = 50
            val_data = []
            train_data = []
            best_val_loss = float("inf")
            for epoch in range(1, num_epochs + 1):

                print(f"Epoch {epoch}/{num_epochs} | Miner: {miner.type_of_triplets}")
                train_mean_loss = train(model, loss, miner, device, train_loader, optimizer, epoch)
                train_data.append({"epoch": epoch, "loss": train_mean_loss})
                val_loss = validate(model, val_loader, loss, device)
                val_data.append({"epoch": epoch, "val_loss": val_loss})
                print(f"Validation Mean Loss on epoch {epoch}: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join("dml_results", backbone_name, "models", f"best_model_state_dict_{miner_type}_epoch-{epoch}.pth"))
                    print("New best model saved!")
                if save_every_epoch:
                    torch.save(model.state_dict(), os.path.join("dml_results", backbone_name, "models", f"model_state_dict_{miner_type}_epoch-{epoch}.pth"))
            print("Training complete")

            # Validation
            final_val_loss = validate(model, val_loader, loss, device)
            print(f"Final Validation Mean Loss: {final_val_loss:.4f}")

            training_info = pd.DataFrame(train_data)

            val_info = pd.DataFrame(val_data)

            # Plot train and val losses
            plt.figure(figsize=(12, 6))
            plt.plot(training_info["epoch"], training_info["loss"], label="Training Loss")
            plt.savefig(os.path.join("dml_results", backbone_name, f"training_loss{miner_type}.png"))

            plt.clf()
            plt.figure(figsize=(12, 6))
            plt.plot(val_info["epoch"], val_info["val_loss"], label="Validation Loss")
            plt.savefig(os.path.join("dml_results", backbone_name, f"validation_loss{miner_type}.png"))

            # Save to .txt
            with open(os.path.join("dml_results", backbone_name, f"validation_loss{miner_type}.txt"), "w") as f:
                f.write(f"Initial Validation Mean Loss: {initial_val_loss:.4f}\n")
                f.write(f"Final Validation Mean Loss: {final_val_loss:.4f}\n")   

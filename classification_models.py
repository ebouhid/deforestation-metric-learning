import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import timm
import matplotlib.pyplot as plt
import os

class ClassificationModel(LightningModule):
    def __init__(self, model_name, log_dir="logs", pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.log_dir = log_dir
        self.save_hyperparameters()
        
        # Load pretrained model and modify the last layer
        if model_name == 'resnet18':
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=False)
            # Strip output layer
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Single output for binary classification
        elif model_name == 'resnet50':
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=False)
            # Strip output layer
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 1)
        elif model_name == 'vgg19':
            self.model = models.vgg19(pretrained=True)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 1)

        # Define binary metrics
        self.f1 = torchmetrics.F1Score(task='binary', threshold=0.5)
        self.recall = torchmetrics.Recall(task='binary', threshold=0.5)
        self.precision = torchmetrics.Precision(task='binary', threshold=0.5)
        self.balanced_acc = torchmetrics.Accuracy(task='binary', average='weighted', threshold=0.5)
        
        self.criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with logits
        self.best_metrics = {}

        # Store metrics for plotting
        self.train_metrics = {'epoch': [], 'f1': [], 'recall': [], 'precision': [], 'balanced_acc': []}
        self.val_metrics = {'epoch': [], 'f1': [], 'recall': [], 'precision': [], 'balanced_acc': []}

    def forward(self, x):
        return self.model(x).squeeze(1)  # Output is a single value per instance

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch.values()
        logits = self(x)
        loss = self.criterion(logits, y.float())
        
        preds = torch.sigmoid(logits) > 0.5
        # Calculate metrics
        f1 = self.f1(preds, y)
        recall = self.recall(preds, y)
        precision = self.precision(preds, y)
        balanced_acc = self.balanced_acc(preds, y)
        
        # Store train metrics
        self.train_metrics['epoch'].append(self.current_epoch)
        self.train_metrics['f1'].append(f1.item())
        self.train_metrics['recall'].append(recall.item())
        self.train_metrics['precision'].append(precision.item())
        self.train_metrics['balanced_acc'].append(balanced_acc.item())

        # Log loss
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch.values()
        logits = self(x)
        val_loss = self.criterion(logits, y.float())
        preds = torch.sigmoid(logits) > 0.5
        
        # Calculate metrics
        f1 = self.f1(preds, y)
        recall = self.recall(preds, y)
        precision = self.precision(preds, y)
        balanced_acc = self.balanced_acc(preds, y)
        
        # Store validation metrics
        self.val_metrics['epoch'].append(self.current_epoch)
        self.val_metrics['f1'].append(f1.item())
        self.val_metrics['recall'].append(recall.item())
        self.val_metrics['precision'].append(precision.item())
        self.val_metrics['balanced_acc'].append(balanced_acc.item())

        # Log metrics
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True)
        self.log('val_balanced_acc', balanced_acc, on_step=False, on_epoch=True)
        
        return {'val_loss': val_loss, 'val_f1': f1, 'val_recall': recall, 'val_precision': precision, 'val_balanced_acc': balanced_acc}

    def test_step(self, batch, batch_idx):
        # For now, same as validation step
        return self.validation_step(batch, batch_idx)

    # def on_validation_epoch_end(self):
    #     # Plot and save metrics
    #     for metric_name in self.train_metrics.keys():
    #         plt.figure()
    #         plt.plot(self.train_metrics[metric_name], label='Train')
    #         plt.plot(self.val_metrics[metric_name], label='Validation')
    #         plt.title(f'{metric_name.capitalize()} over Epochs')
    #         plt.xlabel('Epoch')
    #         plt.ylabel(metric_name.capitalize())
    #         plt.legend()
            
    #         # Save plot
    #         os.makedirs(self.log_dir, exist_ok=True)
    #         plt_path = os.path.join(self.log_dir, f'{self.model_name}_{metric_name}.png')
    #         plt.savefig(plt_path)
    #         plt.close()

    #         # Log the plot to Lightning logs if desired
    #         self.logger.experiment.add_figure(f'{metric_name.capitalize()} plot', plt, global_step=self.current_epoch)
            
    #     # Reset metrics for next epoch
    #     for key in self.train_metrics.keys():
    #         self.train_metrics[key].clear()
    #         self.val_metrics[key].clear()

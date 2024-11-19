# train.py
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from classification_models import ClassificationModel
from dataset import SegmentClassificationDataset
from torch.utils.data import DataLoader


# Set random seed for reproducibility
seed_everything(42)

def run_experiment(data_dir, batch_size=32, num_epochs=10, output_csv='results.csv', checkpoints_dir='checkpoints'):
    # architectures = ['resnet18', 'resnet50', 'vgg16', 'vgg19']
    architectures = ['resnet18', 'resnet50']
    results = []

    for model_name in architectures:
        model = ClassificationModel(model_name=model_name, log_dir=f"logs/{model_name}")
        train_ds = SegmentClassificationDataset(root_dir=data_dir, regions=['x01', 'x02', 'x06', 'x07', 'x09', 'x10'], transform=True)
        val_ds = SegmentClassificationDataset(root_dir=data_dir, regions=['x03', 'x04', 'x08'], transform=False)

        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=23)
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=23)

        
        # Save the best model checkpoint based on validation f1 score
        checkpoint_callback = ModelCheckpoint(monitor='val_f1', mode='max', save_top_k=1, dirpath=f"{checkpoints_dir}/{model_name}", save_last=True)
        
        trainer = Trainer(max_epochs=num_epochs, callbacks=[checkpoint_callback], devices=[1], num_sanity_val_steps=0)
        trainer.fit(model, train_dataloader, val_dataloader)
        
        # Store results
        for train_epoch, train_f1, train_recall, train_precision, train_balanced_acc in zip(model.train_metrics['epoch'], model.train_metrics['f1'], model.train_metrics['recall'], model.train_metrics['precision'], model.train_metrics['balanced_acc']):
            results.append({
                'model': model_name,
                'loop': 'train',
                'epoch': train_epoch,
                'f1': train_f1,
                'recall': train_recall,
                'precision': train_precision,
                'balanced_acc': train_balanced_acc
            })
        
        for val_epoch, val_f1, val_recall, val_precision, val_balanced_acc in zip(model.val_metrics['epoch'], model.val_metrics['f1'], model.val_metrics['recall'], model.val_metrics['precision'], model.val_metrics['balanced_acc']):
            results.append({
                'model': model_name,
                'loop': 'val',
                'epoch': val_epoch,
                'f1': val_f1,
                'recall': val_recall,
                'precision': val_precision,
                'balanced_acc': val_balanced_acc
            })
    
    # Group by train and val epoch and take mean value of each metric
    results = pd.DataFrame.from_records(results)
    results = results.groupby(['model', 'loop', 'epoch']).mean().reset_index()

    results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

data_dir = "./segment_embeddings_classification_dataset_norsz/"
run_experiment(data_dir=data_dir, batch_size=64, num_epochs=200, output_csv='model_metrics_norsz.csv', checkpoints_dir='checkpoints_norsz')

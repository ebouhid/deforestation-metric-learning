# train.py
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from classification_models import ClassificationModel
from dataset import SegmentClassificationDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


# Set random seed for reproducibility
seed_everything(42)

def run_experiment(data_dir, batch_size=32, num_epochs=10, output_csv='results.csv', checkpoints_dir='checkpoints'):
    # architectures = ['resnet18', 'resnet50', 'vgg16', 'vgg19']
    architectures = ['resnet18', 'resnet50']
    results = []

    for model_name in architectures:
        model = ClassificationModel(model_name=model_name, log_dir=f"logs/{model_name}")
        train_ds = SegmentClassificationDataset(root_dir=data_dir, regions=['x02', 'x06', 'x07', 'x09', 'x10'], transform=True)
        val_ds = SegmentClassificationDataset(root_dir=data_dir, regions=['x03', 'x08'], transform=False)
        # Regions x01 and x04 are in the test set

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

        # Evaluate best model on test set
        model = ClassificationModel.load_from_checkpoint(checkpoint_callback.best_model_path)
        test_ds = SegmentClassificationDataset(root_dir=data_dir, regions=['x01', 'x04'], transform=False)
        test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=23)
        trainer.test(model, test_dataloader)

        test_f1 = model.f1.compute()
        test_recall = model.recall.compute()
        test_precision = model.precision.compute()
        test_balanced_acc = model.balanced_acc.compute()

        results.append({
            'model': model_name,
            'loop': 'test',
            'epoch': 0,
            'f1': test_f1,
            'recall': test_recall,
            'precision': test_precision,
            'balanced_acc': test_balanced_acc
        })
    
    # Group by train and val epoch and take mean value of each metric
    results = pd.DataFrame.from_records(results)
    results = results.groupby(['model', 'loop', 'epoch']).mean().reset_index()

    results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    # Plot and save metrics, log to lightning
    for metric_name in model.train_metrics.keys():
        plt.figure()
        for model_name in architectures:
            model_results = results[results['model'] == model_name]
            plt.plot(model_results[model_results['loop'] == 'train']['epoch'], model_results[model_results['loop'] == 'train'][metric_name], label=f'{model_name} Train')
            plt.plot(model_results[model_results['loop'] == 'val']['epoch'], model_results[model_results['loop'] == 'val'][metric_name], label=f'{model_name} Validation')
        plt.title(f'{metric_name.capitalize()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        
        # Create a network_training_results folder in the lightning experiment
        os.makedirs(f"{trainer.logger.log_dir}/network_training_results", exist_ok=True)
        plt_path = os.path.join(trainer.logger.log_dir, "network_training_results", f'{metric_name}.png')
        plt.savefig(plt_path)
        plt.close()


data_dir = "./segment_embeddings_classification_dataset_norsz/"
run_experiment(data_dir=data_dir, batch_size=64, num_epochs=100, output_csv='model_metrics2.csv', checkpoints_dir='network_checkpoints2')

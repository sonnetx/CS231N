'''
This script fine-tunes a ViT model on the ISIC 2019 dataset with various resolutions.
It includes data augmentation, model evaluation, and GPU memory monitoring.
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer
from transformers import AutoModelForImageClassification, AutoImageProcessor  # For DINOv2
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import timm  # For SimCLR's ResNet backbone

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("pynvml not installed, GPU memory monitoring disabled.")
from thop import profile

# Custom Dataset for ISIC 2019 with Downsampling and Model-Specific Preprocessing
class ISICDataset(Dataset):
    def __init__(self, dataset, preprocessor, resolution=224, transform=None, model_type='vit'):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.resolution = resolution
        self.transform = transform
        self.model_type = model_type

        # SimCLR-specific preprocessing
        if model_type == 'simclr':
            self.preprocessor = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']  # PIL Image
        label = item['label']  # Integer label (0-7)
        
        # Downsample image
        if self.resolution != 224:
            image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        if self.transform:
            image = self.transform(image)
        
        # Preprocess based on model type
        if self.model_type in ['vit', 'dinov2']:
            encoding = self.preprocessor(images=image, return_tensors="pt")
            pixel_values = encoding['pixel_values'].squeeze(0)
        elif self.model_type == 'simclr':
            pixel_values = self.preprocessor(image)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        label = torch.tensor(label, dtype=torch.long)
        
        return {'pixel_values': pixel_values, 'labels': label}

# Compute metrics for evaluation
def compute_metrics(eval_pred, model_name, resolution):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    auc = roc_auc_score(labels, logits, multi_class='ovr')
    
    # Plot confusion matrix
    conf_mat = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'{model_name}_{resolution}_conf_mat')
    plt.savefig(f'{model_name}_{resolution}_conf_mat.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Classification breakdown
    unique, counts = np.unique(predictions, return_counts=True)
    class_breakdown = dict(zip(unique, counts))
    with open(f'{model_name}_{resolution}_class_breakdown.json', 'w') as f:
        json.dump(class_breakdown, f)
    
    return {'accuracy': acc, 'f1': f1, 'auc': auc}

# Measure GPU memory usage
def get_gpu_memory(device_id=0):
    if not GPU_AVAILABLE:
        return -1
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1024**2  # MB
    except:
        return -1
    
# Wrapper for SimCLR to match Trainer API
class SimCLRForClassification(nn.Module):
    def __init__(self, backbone, num_classes=8):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2048, num_classes)  # ResNet-50 output: 2048

    def forward(self, pixel_values, labels=None):
        features = self.backbone(pixel_values).pooler_output  # Get pooled output
        logits = self.classifier(features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'logits': logits, 'loss': loss} if loss is not None else {'logits': logits}

# Main function for fine-tuning
def main():
    # Models and resolutions to compare
    models = [
        {'name': 'vit', 'model_id': 'google/vit-base-patch16-224', 'type': 'vit'},
        {'name': 'dinov2', 'model_id': 'facebook/dinov2-base', 'type': 'dinov2'},  # Load from Hugging Face
        {'name': 'simclr', 'model_id': 'resnet50', 'type': 'simclr'},  # ResNet-50 from timm
    ]
    resolutions = [224, 112, 56]
    
    # Results storage
    results = {model['name']: {} for model in models}
    
    # Load dataset
    dataset = load_dataset("MKZuziak/ISIC_2019_224")
    full_dataset = dataset['train'].train_test_split(test_size=0.2, stratify_by_column='label', seed=42)
    train_dataset = full_dataset['train']
    val_dataset = full_dataset['test']
    
    # Data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    
    for model_info in models:
        model_name = model_info['name']
        model_id = model_info['model_id']
        model_type = model_info['type']
        print(f"\nFine-tuning model: {model_name}")
        
        for resolution in resolutions:
            print(f"Resolution: {resolution}x{resolution}")
            
            # Load preprocessor
            if model_type == 'vit':
                preprocessor = ViTFeatureExtractor.from_pretrained(model_id, size=resolution)
            elif model_type == 'dinov2':
                preprocessor = AutoImageProcessor.from_pretrained(model_id, size=resolution)
            elif model_type == 'simclr':
                preprocessor = None  # Handled in ISICDataset
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
            
            # Create datasets
            train_ds = ISICDataset(train_dataset, preprocessor, resolution, transform, model_type)
            val_ds = ISICDataset(val_dataset, preprocessor, resolution, model_type=model_type)
            
            # Load model
            if model_type == 'vit':
                model = ViTForImageClassification.from_pretrained(
                    model_id,
                    num_labels=8,
                    ignore_mismatched_sizes=True
                )
            elif model_type == 'dinov2':
                model = AutoModelForImageClassification.from_pretrained(
                    model_id,
                    num_labels=8,
                    ignore_mismatched_sizes=True
                )
            elif model_type == 'simclr':
                # Load ResNet-50 from timm (fallback if no SimCLR weights)
                backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)  # No classifier
                model = SimCLRForClassification(backbone, num_classes=8)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            # Estimate FLOPs
            input_tensor = torch.randn(1, 3, resolution, resolution)
            try:
                flops, _ = profile(model, inputs=(input_tensor,))
                flops = flops / 1e9  # GFLOPs
            except:
                flops = -1  # Fallback if FLOPs estimation fails
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f'./results_{model_name}_{resolution}',
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f'./logs_{model_name}_{resolution}',
                logging_steps=10,
                evaluation_strategy='epoch',
                save_strategy='epoch',
                load_best_model_at_end=True,
                metric_for_best_model='accuracy',
            )
            
            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=lambda pred: compute_metrics(pred, model_name, resolution),
            )
            
            # Measure memory and time
            start_time = time.time()
            peak_memory = get_gpu_memory() if GPU_AVAILABLE else -1
            
            # Fine-tune
            trainer.train()
            
            # Update peak memory
            current_memory = get_gpu_memory() if GPU_AVAILABLE else -1
            peak_memory = max(peak_memory, current_memory)
            
            # Evaluate
            eval_start_time = time.time()
            eval_results = trainer.evaluate()
            eval_time = time.time() - eval_start_time
            
            # Total training time
            train_time = time.time() - start_time - eval_time
            
            # Save model
            model.save_pretrained(f'./finetuned_{model_name}_{resolution}')
            if model_type in ['vit', 'dinov2']:
                preprocessor.save_pretrained(f'./finetuned_{model_name}_{resolution}')
            
            # Store results
            results[model_name][resolution] = {
                'peak_memory_mb': peak_memory,
                'flops_giga': flops,
                'train_time_seconds': train_time,
                'eval_time_seconds': eval_time,
                'eval_metrics': eval_results
            }
            print(f"Results for {model_name} at {resolution}x{resolution}: {results[model_name][resolution]}")
    
    # Save results to JSON
    with open('results_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
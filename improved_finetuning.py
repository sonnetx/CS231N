import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer
import timm
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import numpy as np
import time
import json
import os
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("pynvml not installed, GPU memory monitoring disabled.")
from thop import profile
import torch.nn.functional as F

# Custom Dataset for ISIC 2019
class ISICDataset(Dataset):
    def __init__(self, dataset, preprocessor, resolution=224, transform=None, model_type='vit'):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.resolution = resolution
        self.transform = transform
        self.model_type = model_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        if self.resolution != 224:
            image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        if self.transform:
            image = self.transform(image)
        
        if self.model_type == 'vit':
            encoding = self.preprocessor(images=image, return_tensors="pt")
            pixel_values = encoding['pixel_values'].squeeze(0)
        else:
            image_tensor = transforms.ToTensor()(image)
            image_tensor = transforms.Resize((self.resolution, self.resolution))(image_tensor)
            image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
            pixel_values = image_tensor
        
        label = torch.tensor(label, dtype=torch.long)
        return {'pixel_values': pixel_values, 'labels': label}

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# Class-weighted loss and CutMix trainer
class CustomTrainer(Trainer):
    def __init__(self, use_weighted_loss=False, use_cutmix=False, dataset=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weighted_loss = use_weighted_loss
        self.use_cutmix = use_cutmix
        self.class_weights = get_class_weights(dataset).to(self.args.device) if use_weighted_loss else None
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        pixel_values = inputs['pixel_values']
        
        if self.use_cutmix and np.random.rand() < 0.5:
            pixel_values, labels_a, labels_b, lam = cutmix(pixel_values, labels)
            outputs = model(pixel_values)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = lam * F.cross_entropy(logits, labels_a, weight=self.class_weights) + \
                   (1 - lam) * F.cross_entropy(logits, labels_b, weight=self.class_weights)
        else:
            outputs = model(pixel_values)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        
        return (loss, outputs) if return_outputs else loss

# Distillation trainer
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.to(self.args.device)
        self.teacher_model.eval()
        self.temperature = 2.0
        self.alpha = 0.5
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs
        
        loss_ce = F.cross_entropy(student_logits, labels)
        loss_kl = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kl
        
        return (loss, student_outputs) if return_outputs else loss

# CutMix implementation
def cutmix(data, labels, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]
    
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))
    
    return data, labels, shuffled_labels, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# Class weights
def get_class_weights(dataset):
    labels = [item['label'] for item in dataset]
    counter = Counter(labels)
    total = len(labels)
    weights = {i: total / (len(counter) * count) for i, count in counter.items()}
    return torch.tensor([weights[i] for i in range(8)], dtype=torch.float)

# Weighted sampler for oversampling
def get_sampler(dataset):
    labels = [item['label'] for item in dataset]
    counter = Counter(labels)
    total = len(labels)
    weights = [total / (len(counter) * counter[label]) for label in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

# Test-Time Augmentation
def tta_predict(model, preprocessor, image, resolution, model_type, device='cuda', n_augs=5):
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Resize((resolution, resolution)),
    ])
    model.eval()
    logits = []
    for _ in range(n_augs):
        aug_image = aug_transform(image)
        if model_type == 'vit':
            encoding = preprocessor(images=aug_image, return_tensors="pt")
            pixel_values = encoding['pixel_values'].to(device)
        else:
            image_tensor = transforms.ToTensor()(aug_image)
            image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
            pixel_values = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values)
            logits.append(outputs.logits if model_type == 'vit' else outputs)
    
    avg_logits = torch.mean(torch.stack(logits), dim=0)
    return torch.softmax(avg_logits, dim=-1)

def evaluate_tta(model, preprocessor, val_dataset, resolution, model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions, true_labels = [], []
    model.to(device)
    for item in val_dataset:
        image = item['image']
        label = item['label']
        probs = tta_predict(model, preprocessor, image, resolution, model_type, device)
        pred = torch.argmax(probs, dim=-1).cpu().numpy()
        predictions.append(pred)
        true_labels.append(label)
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# Ensemble prediction
def load_model(model_name, model_id, model_type, resolution, num_labels=8):
    if model_type == 'vit':
        return ViTForImageClassification.from_pretrained(f'./finetuned_{model_name}_{resolution}')
    else:
        return timm.create_model(model_name, num_classes=num_labels, checkpoint_path=f'./finetuned_{model_name}_{resolution}/pytorch_model.bin')

def ensemble_predict(models, preprocessors, image, resolution, device='cuda'):
    image = image.resize((resolution, resolution), Image.LANCZOS)
    logits = []
    for model_info, preprocessor in zip(models, preprocessors):
        model_name = model_info['name']
        model_type = model_info['type']
        model = load_model(model_name, model_info['model_id'], model_type, resolution).to(device).eval()
        
        if model_type == 'vit':
            encoding = preprocessor(images=image, return_tensors="pt")
            pixel_values = encoding['pixel_values'].to(device)
        else:
            image_tensor = transforms.ToTensor()(image)
            image_tensor = transforms.Resize((resolution, resolution))(image_tensor)
            image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
            pixel_values = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values)
            logits.append(outputs.logits if model_type == 'vit' else outputs)
    
    avg_logits = torch.mean(torch.stack(logits), dim=0)
    return torch.softmax(avg_logits, dim=-1)

def evaluate_ensemble(models, preprocessors, val_dataset, resolution):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions, true_labels = [], []
    for item in val_dataset:
        image = item['image']
        label = item['label']
        probs = ensemble_predict(models, preprocessors, image, resolution, device)
        pred = torch.argmax(probs, dim=-1).cpu().numpy()
        predictions.append(pred)
        true_labels.append(label)
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# GPU memory usage
def get_gpu_memory(device_id=0):
    if not GPU_AVAILABLE:
        return -1
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1024**2  # MB
    except:
        return -1

# Main function
def main():
    # Models, resolutions, and advancements
    models = [
        {'name': 'vit', 'model_id': 'google/vit-base-patch16-224', 'type': 'vit'},
        {'name': 'resnet50', 'model_id': 'resnet50', 'type': 'timm'},
        {'name': 'efficientnet_b0', 'model_id': 'efficientnet_b0', 'type': 'timm'}
    ]
    resolutions = [224, 112, 56]
    advancements = [
        'baseline',
        'weighted_loss',
        'oversampling',
        'advanced_augmentation',
        'tta',  # Evaluated post-training
        'ensemble',  # Evaluated post-training
        'distillation'  # Separate loop for EfficientNet-B0
    ]
    
    # Results storage
    results = {adv: {model['name']: {} for model in models} for adv in advancements}
    results['ensemble'] = {}  # Separate for ensemble
    results['distillation'] = {}  # Separate for distilled EfficientNet
    
    # Load dataset
    dataset = load_dataset("MKZuziak/ISIC_2019_224")
    full_dataset = dataset['train'].train_test_split(test_size=0.2, stratify_by_column='label', seed=42)
    train_dataset = full_dataset['train']
    val_dataset = full_dataset['test']
    
    # Transforms
    basic_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    advanced_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    ])
    
    for advancement in advancements:
        if advancement in ['ensemble', 'distillation', 'tta']:
            continue  # Handled separately
        
        print(f"\nRunning advancement: {advancement}")
        
        for model_info in models:
            model_name = model_info['name']
            model_id = model_info['model_id']
            model_type = model_info['type']
            print(f"Model: {model_name}")
            
            for resolution in resolutions:
                print(f"Resolution: {resolution}x{resolution}")
                
                # Preprocessor
                preprocessor = ViTFeatureExtractor.from_pretrained(model_id, size=resolution) if model_type == 'vit' else None
                
                # Transform
                transform = advanced_transform if advancement == 'advanced_augmentation' else basic_transform
                
                # Dataset
                train_ds = ISICDataset(train_dataset, preprocessor, resolution, transform, model_type)
                val_ds = ISICDataset(val_dataset, preprocessor, resolution, model_type=model_type)
                
                # Sampler for oversampling
                sampler = get_sampler(train_ds) if advancement == 'oversampling' else None
                
                # Model
                model = ViTForImageClassification.from_pretrained(model_id, num_labels=8, ignore_mismatched_sizes=True) if model_type == 'vit' else \
                        timm.create_model(model_id, pretrained=True, num_classes=8)
                
                # FLOPs
                input_tensor = torch.randn(1, 3, resolution, resolution)
                try:
                    flops, _ = profile(model, inputs=(input_tensor,))
                    flops = flops / 1e9
                except:
                    flops = -1
                
                # Training arguments
                training_args = TrainingArguments(
                    output_dir=f'./results_{advancement}_{model_name}_{resolution}',
                    num_train_epochs=3,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    warmup_steps=500,
                    weight_decay=0.01,
                    logging_dir=f'./logs_{advancement}_{model_name}_{resolution}',
                    logging_steps=10,
                    evaluation_strategy='epoch',
                    save_strategy='epoch',
                    load_best_model_at_end=True,
                    metric_for_best_model='accuracy',
                )
                
                # Trainer
                trainer = CustomTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=val_ds,
                    compute_metrics=compute_metrics,
                    use_weighted_loss=(advancement == 'weighted_loss'),
                    use_cutmix=(advancement == 'advanced_augmentation'),
                    dataset=train_dataset
                )
                
                # Train
                start_time = time.time()
                peak_memory = get_gpu_memory() if GPU_AVAILABLE else -1
                trainer.train()
                current_memory = get_gpu_memory() if GPU_AVAILABLE else -1
                peak_memory = max(peak_memory, current_memory)
                
                # Evaluate
                eval_start_time = time.time()
                eval_results = trainer.evaluate()
                eval_time = time.time() - eval_start_time
                train_time = time.time() - start_time - eval_time
                
                # TTA (if applicable)
                tta_metrics = evaluate_tta(model, preprocessor, val_dataset, resolution, model_type) if advancement == 'tta' else None
                
                # Save model
                model.save_pretrained(f'./finetuned_{advancement}_{model_name}_{resolution}')
                if model_type == 'vit':
                    preprocessor.save_pretrained(f'./finetuned_{advancement}_{model_name}_{resolution}')
                
                # Store results
                results[advancement][model_name][resolution] = {
                    'peak_memory_mb': peak_memory,
                    'flops_giga': flops,
                    'train_time_seconds': train_time,
                    'eval_time_seconds': eval_time,
                    'eval_metrics': eval_results
                }
                if tta_metrics:
                    results[advancement][model_name][resolution]['tta_metrics'] = tta_metrics
                print(f"Results for {advancement}, {model_name}, {resolution}x{resolution}: {results[advancement][model_name][resolution]}")
    
    # Ensemble
    for resolution in resolutions:
        print(f"\nRunning ensemble for resolution: {resolution}x{resolution}")
        preprocessors = []
        for model_info in models:
            model_name = model_info['name']
            model_type = model_info['type']
            if model_type == 'vit':
                preprocessors.append(ViTFeatureExtractor.from_pretrained(f'./finetuned_baseline_{model_name}_{resolution}'))
            else:
                preprocessors.append(None)
        ensemble_metrics = evaluate_ensemble(models, preprocessors, val_dataset, resolution)
        results['ensemble'][resolution] = {
            'eval_metrics': ensemble_metrics,
            'peak_memory_mb': -1,
            'flops_giga': -1,
            'train_time_seconds': -1,
            'eval_time_seconds': -1
        }
        print(f"Ensemble results at {resolution}x{resolution}: {ensemble_metrics}")
    
    # Knowledge Distillation
    for resolution in resolutions:
        print(f"\nRunning distillation for resolution: {resolution}x{resolution}")
        teacher_model = load_model('vit', models[0]['model_id'], 'vit', resolution)
        student_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=8)
        
        preprocessor = ViTFeatureExtractor.from_pretrained(f'./finetuned_baseline_vit_{resolution}', size=resolution)
        train_ds = ISICDataset(train_dataset, preprocessor, resolution, basic_transform, model_type='vit')
        val_ds = ISICDataset(val_dataset, preprocessor, resolution, model_type='vit')
        
        training_args = TrainingArguments(
            output_dir=f'./results_distillation_efficientnet_{resolution}',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'./logs_distillation_efficientnet_{resolution}',
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )
        
        trainer = DistillationTrainer(
            teacher_model=teacher_model,
            model=student_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        
        start_time = time.time()
        peak_memory = get_gpu_memory() if GPU_AVAILABLE else -1
        trainer.train()
        current_memory = get_gpu_memory() if GPU_AVAILABLE else -1
        peak_memory = max(peak_memory, current_memory)
        
        eval_start_time = time.time()
        eval_results = trainer.evaluate()
        eval_time = time.time() - eval_start_time
        train_time = time.time() - start_time - eval_time
        
        input_tensor = torch.randn(1, 3, resolution, resolution)
        try:
            flops, _ = profile(student_model, inputs=(input_tensor,))
            flops = flops / 1e9
        except:
            flops = -1
        
        results['distillation'][resolution] = {
            'peak_memory_mb': peak_memory,
            'flops_giga': flops,
            'train_time_seconds': train_time,
            'eval_time_seconds': eval_time,
            'eval_metrics': eval_results
        }
        student_model.save_pretrained(f'./finetuned_distillation_efficientnet_{resolution}')
        print(f"Distillation results at {resolution}x{resolution}: {results['distillation'][resolution]}")
    
    # Save results
    with open('results_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()